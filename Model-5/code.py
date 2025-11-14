"""
HRNet-W48 with Hybrid Edge Channels Training Script for CAID Dataset
Semantic Segmentation: Land vs Water Classification
With Modified mSCR Evaluation (Per-k Averaging) and Enhanced Edge Detection
"""

import os
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import logging
from datetime import datetime
from IPython.display import display, clear_output, Image as IPImage
from scipy.ndimage import binary_dilation
import io
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import torchvision.transforms as T

plt.ion()

# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logger(log_dir):
    """Setup comprehensive logging system"""
    os.makedirs(log_dir, exist_ok=True)

    # Create log filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'training_{timestamp}.log')

    # Configure logger
    logger = logging.getLogger('HRNet_CAID')
    logger.setLevel(logging.DEBUG)

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s | %(name)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    # Paths (Kaggle environment)
    DATA_ROOT = '/kaggle/input/caid-dataset/CAID'
    IMAGES_DIR = os.path.join(DATA_ROOT, 'JPEGImages')
    MASKS_DIR = os.path.join(DATA_ROOT, 'SegmentationClass')
    TRAIN_TXT = os.path.join(DATA_ROOT, 'ImageSets/Segmentation/train.txt')
    VAL_TXT = os.path.join(DATA_ROOT, 'ImageSets/Segmentation/val.txt')
    TEST_TXT = os.path.join(DATA_ROOT, 'ImageSets/Segmentation/test.txt')

    # Output
    OUTPUT_DIR = '/kaggle/working'
    LOG_DIR = os.path.join(OUTPUT_DIR, 'logs')
    BEST_MODEL_PATH = os.path.join(OUTPUT_DIR, 'hrnet_hybrid_best.pth')

    # Training hyperparameters
    NUM_CLASSES = 2  # Land (0) and Water (1)
    BATCH_SIZE = 8  # Reduced for HRNet-W48
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4

    # Model
    BACKBONE = 'hrnet_w48'
    PRETRAINED = True

    # Hybrid input settings
    USE_EDGE_CHANNELS = True  # Enable 5-channel input (RGB + Sobel + Canny)
    INPUT_CHANNELS = 5 if USE_EDGE_CHANNELS else 3

    # Edge detection parameters
    SOBEL_KSIZE = 3
    CANNY_THRESHOLD1 = 50
    CANNY_THRESHOLD2 = 150

    # Training options
    USE_MIXED_PRECISION = True
    NUM_WORKERS = 2
    PIN_MEMORY = True

    # Image settings
    IMAGE_SIZE = (500, 500)

    # Augmentation probabilities
    FLIP_PROB = 0.5
    ROTATE_PROB = 0.3

    # mSCR evaluation settings - Per-k averaging
    K_NEIGHBORS = [4, 8, 24, 48]

    # Loss weights
    LOSS_WEIGHT_CE = 1.0      # Cross-entropy loss weight
    LOSS_WEIGHT_DICE = 1.0    # Dice loss weight
    LOSS_WEIGHT_MSCR = 0.5    # mSCR loss weight (differentiable)

    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

cfg = Config()

# Initialize logger
logger = setup_logger(cfg.LOG_DIR)

# ============================================================================
# EDGE DETECTION UTILITIES
# ============================================================================

def compute_sobel_magnitude(image_np):
    """
    Compute Sobel edge magnitude from RGB image.

    Args:
        image_np: numpy array (H, W, 3) in range [0, 255]

    Returns:
        Sobel magnitude normalized to [0, 1]
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    # Compute Sobel gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=cfg.SOBEL_KSIZE)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=cfg.SOBEL_KSIZE)

    # Compute magnitude
    magnitude = np.sqrt(sobelx**2 + sobely**2)

    # Normalize to [0, 1]
    if magnitude.max() > 0:
        magnitude = magnitude / magnitude.max()

    return magnitude.astype(np.float32)


def compute_canny_edges(image_np):
    """
    Compute Canny edges from RGB image.

    Args:
        image_np: numpy array (H, W, 3) in range [0, 255]

    Returns:
        Binary edge map normalized to [0, 1]
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    # Apply Canny edge detection
    edges = cv2.Canny(gray, cfg.CANNY_THRESHOLD1, cfg.CANNY_THRESHOLD2)

    # Normalize to [0, 1]
    edges = edges.astype(np.float32) / 255.0

    return edges


def create_hybrid_input(image):
    """
    Create 5-channel hybrid input: [R, G, B, Sobel, Canny]

    Args:
        image: PIL Image

    Returns:
        numpy array (H, W, 5)
    """
    # Convert PIL to numpy
    image_np = np.array(image)

    # Compute edge channels
    sobel = compute_sobel_magnitude(image_np)
    canny = compute_canny_edges(image_np)

    # Stack channels
    hybrid = np.dstack([
        image_np[:, :, 0],  # R
        image_np[:, :, 1],  # G
        image_np[:, :, 2],  # B
        sobel,              # Sobel magnitude
        canny               # Canny edges
    ])

    return hybrid.astype(np.float32)

# ============================================================================
# DATASET CLASS WITH HYBRID INPUT
# ============================================================================

class CAIDHybridDataset(Dataset):
    """
    Custom Dataset for CAID with Hybrid Edge Channels
    Loads images and creates 5-channel input (RGB + Sobel + Canny)
    """

    def __init__(self, split_file, images_dir, masks_dir, transform=None, augment=False):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.augment = augment

        # Read image IDs from split file
        with open(split_file, 'r') as f:
            self.image_ids = [line.strip() for line in f.readlines()]

        logger.info(f"Loaded {len(self.image_ids)} samples from {split_file}")

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        # Get image ID
        img_id = self.image_ids[idx]

        # Load image and mask
        img_path = os.path.join(self.images_dir, f"{img_id}.png")
        mask_path = os.path.join(self.masks_dir, f"{img_id}.png")

        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        # Apply augmentations (synchronized for image and mask)
        if self.augment:
            image, mask = self.apply_augmentation(image, mask)

        # Create hybrid input (5 channels)
        if cfg.USE_EDGE_CHANNELS:
            hybrid_input = create_hybrid_input(image)  # (H, W, 5)

            # Convert to tensor and normalize
            # RGB channels: ImageNet normalization
            # Edge channels: already in [0, 1], normalize to [-1, 1]
            rgb = hybrid_input[:, :, :3] / 255.0  # [0, 1]
            edges = hybrid_input[:, :, 3:]  # Already [0, 1]

            # Apply ImageNet normalization to RGB
            rgb_normalized = (rgb - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])

            # Normalize edges to [-1, 1]
            edges_normalized = edges * 2.0 - 1.0

            # Concatenate
            final_input = np.concatenate([rgb_normalized, edges_normalized], axis=2)

            # Convert to tensor (H, W, C) -> (C, H, W)
            image_tensor = torch.from_numpy(final_input).permute(2, 0, 1).float()
        else:
            # Standard 3-channel processing
            image_tensor = T.ToTensor()(image)
            image_tensor = T.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])(image_tensor)

        # Convert mask to tensor
        mask = torch.from_numpy(np.array(mask)).long()

        return image_tensor, mask

    def apply_augmentation(self, image, mask):
        """Apply synchronized augmentations to image and mask"""

        # Random horizontal flip
        if np.random.rand() < cfg.FLIP_PROB:
            image = T.functional.hflip(image)
            mask = T.functional.hflip(mask)

        # Random vertical flip
        if np.random.rand() < cfg.FLIP_PROB:
            image = T.functional.vflip(image)
            mask = T.functional.vflip(mask)

        # Random rotation (90, 180, 270 degrees)
        if np.random.rand() < cfg.ROTATE_PROB:
            angle = int(np.random.choice([90, 180, 270]))
            image = T.functional.rotate(image, angle)
            mask = T.functional.rotate(mask, angle)

        return image, mask

# ============================================================================
# HRNET-W48 MODEL IMPLEMENTATION
# ============================================================================

class BasicBlock(nn.Module):
    """Basic residual block for HRNet"""
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


# ============================================================================
# ATTENTION MECHANISMS AND BOUNDARY MODULES
# ============================================================================

class ChannelAttention(nn.Module):
    """
    Channel Attention Module (Squeeze-and-Excitation style)
    Helps model learn importance of different channels (RGB vs Edge channels)
    """
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        attention = self.sigmoid(avg_out + max_out)
        return x * attention


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module
    Focuses on important spatial regions (boundaries/shorelines)
    """
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(spatial_input))
        return x * attention


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module
    Combines channel and spatial attention sequentially
    """
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)
    
    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class BoundaryRefinementModule(nn.Module):
    """
    Boundary Refinement Module
    Explicitly refines boundary predictions using edge-aware convolutions
    """
    def __init__(self, in_channels):
        super(BoundaryRefinementModule, self).__init__()
        
        # Edge-aware convolutions
        self.edge_conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True)
        )
        
        self.edge_conv2 = nn.Sequential(
            nn.Conv2d(in_channels // 2, in_channels // 2, 3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True)
        )
        
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels + in_channels // 2, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        edge1 = self.edge_conv1(x)
        edge2 = self.edge_conv2(edge1)
        refined = self.fusion(torch.cat([x, edge2], dim=1))
        return refined


class EnhancedMultiScaleFusion(nn.Module):
    """
    Enhanced Multi-Scale Feature Fusion
    Better aggregates features from different HRNet branches with attention
    """
    def __init__(self, channels_list=[48, 96, 192, 384], out_channels=256):
        super(EnhancedMultiScaleFusion, self).__init__()
        
        # Individual processing for each scale
        self.scale_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, out_channels // 4, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels // 4),
                nn.ReLU(inplace=True),
                CBAM(out_channels // 4, reduction=8)
            ) for ch in channels_list
        ])
        
        # Fusion with attention
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            CBAM(out_channels, reduction=16)
        )
        
        # Boundary refinement
        self.boundary_refine = BoundaryRefinementModule(out_channels)
    
    def forward(self, features_list, target_size):
        """
        Args:
            features_list: List of features from different branches [feat1, feat2, feat3, feat4]
            target_size: Target spatial size (H, W) for upsampling
        """
        processed_features = []
        
        for i, (feat, conv) in enumerate(zip(features_list, self.scale_convs)):
            # Upsample to target size
            feat_up = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)
            # Process with attention
            feat_processed = conv(feat_up)
            processed_features.append(feat_processed)
        
        # Concatenate all scales
        fused = torch.cat(processed_features, dim=1)
        
        # Apply fusion convolution with attention
        fused = self.fusion_conv(fused)
        
        # Boundary refinement
        refined = self.boundary_refine(fused)
        
        return refined


# ============================================================================
# ENHANCED HRNET-W48 MODEL WITH ATTENTION
# ============================================================================

class HRNetW48(nn.Module):
    """
    Enhanced HRNet-W48 with Attention Mechanisms for Boundary-Focused Segmentation
    """

    def __init__(self, num_classes=2, input_channels=5):
        super(HRNetW48, self).__init__()

        # Stage 1: Stem with Channel Attention for hybrid input
        self.conv1 = nn.Conv2d(input_channels, 64, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 3, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        # Input channel attention - learn to weight RGB vs edge channels
        self.input_attention = ChannelAttention(64, reduction=8)

        # Stage 1: High-resolution branch
        self.layer1 = self._make_layer(BasicBlock, 64, 64, 4)

        # Transition to parallel branches (stage 2)
        # Note: layer1 outputs 64 channels (not 256), so we use 64 here
        self.transition1 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(64, 48, 3, 1, 1, bias=False),
                nn.BatchNorm2d(48),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(64, 96, 3, 2, 1, bias=False),
                nn.BatchNorm2d(96),
                nn.ReLU(inplace=True)
            )
        ])

        # Stage 2
        self.stage2 = nn.ModuleList([
            self._make_layer(BasicBlock, 48, 48, 4),
            self._make_layer(BasicBlock, 96, 96, 4)
        ])

        # Transition to stage 3
        self.transition2 = nn.ModuleList([
            None,
            None,
            nn.Sequential(
                nn.Conv2d(96, 192, 3, 2, 1, bias=False),
                nn.BatchNorm2d(192),
                nn.ReLU(inplace=True)
            )
        ])

        # Stage 3
        self.stage3 = nn.ModuleList([
            self._make_layer(BasicBlock, 48, 48, 4),
            self._make_layer(BasicBlock, 96, 96, 4),
            self._make_layer(BasicBlock, 192, 192, 4)
        ])

        # Transition to stage 4
        self.transition3 = nn.ModuleList([
            None,
            None,
            None,
            nn.Sequential(
                nn.Conv2d(192, 384, 3, 2, 1, bias=False),
                nn.BatchNorm2d(384),
                nn.ReLU(inplace=True)
            )
        ])

        # Stage 4
        self.stage4 = nn.ModuleList([
            self._make_layer(BasicBlock, 48, 48, 4),
            self._make_layer(BasicBlock, 96, 96, 4),
            self._make_layer(BasicBlock, 192, 192, 4),
            self._make_layer(BasicBlock, 384, 384, 4)
        ])
        
        # Add attention to each stage output
        self.stage4_attention = nn.ModuleList([
            CBAM(48, reduction=8),
            CBAM(96, reduction=8),
            CBAM(192, reduction=8),
            CBAM(384, reduction=8)
        ])

        # Enhanced multi-scale fusion with attention
        self.multi_scale_fusion = EnhancedMultiScaleFusion(
            channels_list=[48, 96, 192, 384],
            out_channels=256
        )
        
        # Main segmentation head
        self.seg_head = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(128, num_classes, 1)
        )
        
        # Auxiliary boundary detection head (REMOVED sigmoid for autocast compatibility)
        self.boundary_head = nn.Sequential(
            nn.Conv2d(256, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1)
            # Sigmoid removed - will use BCEWithLogitsLoss instead
        )

        self._init_weights()

    def _make_layer(self, block, inplanes, planes, blocks):
        """Create a residual layer"""
        layers = []
        layers.append(block(inplanes, planes))
        for _ in range(1, blocks):
            layers.append(block(planes, planes))
        return nn.Sequential(*layers)

    def _init_weights(self):
        """Initialize weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        input_size = x.size()[2:]

        # Stem with input attention
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.input_attention(x)  # Learn to weight RGB vs edge channels
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        # Stage 1
        x = self.layer1(x)

        # Transition to stage 2
        x_list = []
        for i, trans in enumerate(self.transition1):
            x_list.append(trans(x))

        # Stage 2
        y_list = []
        for i, stage in enumerate(self.stage2):
            y_list.append(stage(x_list[i]))

        # Transition to stage 3
        x_list = []
        for i, trans in enumerate(self.transition2):
            if trans is None:
                x_list.append(y_list[i])
            else:
                x_list.append(trans(y_list[i-1]))

        # Stage 3
        y_list = []
        for i, stage in enumerate(self.stage3):
            y_list.append(stage(x_list[i]))

        # Transition to stage 4
        x_list = []
        for i, trans in enumerate(self.transition3):
            if trans is None:
                x_list.append(y_list[i])
            else:
                x_list.append(trans(y_list[i-1]))

        # Stage 4 with attention
        y_list = []
        for i, (stage, attention) in enumerate(zip(self.stage4, self.stage4_attention)):
            feat = stage(x_list[i])
            feat = attention(feat)  # Apply CBAM attention
            y_list.append(feat)

        # Enhanced multi-scale fusion
        target_size = (y_list[0].size(2), y_list[0].size(3))
        fused_features = self.multi_scale_fusion(y_list, target_size)
        
        # Segmentation output
        seg_out = self.seg_head(fused_features)
        
        # Boundary prediction (auxiliary)
        boundary_out = self.boundary_head(fused_features)
        
        # Upsample to input size
        seg_out = F.interpolate(seg_out, size=input_size, mode='bilinear', align_corners=False)
        boundary_out = F.interpolate(boundary_out, size=input_size, mode='bilinear', align_corners=False)

        return seg_out, boundary_out

# ============================================================================
# MODIFIED mSCR METRIC EVALUATION (Per-k Averaging)
# ============================================================================

def extract_shoreline_pixels(mask):
    """Extract shoreline (boundary) pixels from a binary mask"""
    binary_mask = (mask > 0).astype(np.uint8)

    dilated = binary_dilation(binary_mask, structure=np.ones((3, 3)))
    eroded = binary_dilation(1 - binary_mask, structure=np.ones((3, 3)))

    shoreline = dilated & eroded

    return shoreline


def expand_neighborhood(shoreline_pixels, k):
    """Expand shoreline pixels by k-neighborhood"""
    if k == 4:
        structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
    elif k == 8:
        structure = np.ones((3, 3), dtype=np.uint8)
    elif k == 24:
        structure = np.ones((5, 5), dtype=np.uint8)
    elif k == 48:
        structure = np.ones((7, 7), dtype=np.uint8)
    else:
        raise ValueError(f"Unsupported k value: {k}")

    expanded = binary_dilation(shoreline_pixels.astype(np.uint8), structure=structure)

    return expanded


def compute_scr_single(shoreline_1, shoreline_2_expanded):
    """Compute SCR for a single direction"""
    S1 = np.sum(shoreline_1)
    S2_expanded = np.sum(shoreline_2_expanded)

    if S1 == 0 and S2_expanded == 0:
        return 1.0
    if S1 == 0 or S2_expanded == 0:
        return 0.0

    intersection = np.sum(shoreline_1 & shoreline_2_expanded)
    scr = (2.0 * intersection) / (S1 + S2_expanded)

    return scr


def compute_scr_by_k_for_batch(pred_batch, target_batch, k_values=[4, 8, 24, 48]):
    """
    MODIFIED: Computes SCR for a batch, averaging across IMAGES for EACH k-value.

    Returns a dictionary with mean scores for each k:
    {'SCR_k4_mean': 0.85, 'SCR_k8_mean': 0.88, ...}
    """
    # Convert to numpy if tensor
    if torch.is_tensor(pred_batch):
        pred_batch = pred_batch.cpu().numpy()
    if torch.is_tensor(target_batch):
        target_batch = target_batch.cpu().numpy()

    batch_size = pred_batch.shape[0]

    # Dictionary to hold scores for each k-value
    scr_scores_by_k = {k: [] for k in k_values}

    # Loop through each image in the batch
    for i in range(batch_size):
        pred_shoreline = extract_shoreline_pixels(pred_batch[i])
        gt_shoreline = extract_shoreline_pixels(target_batch[i])

        # Calculate SCR for each k and store it
        for k in k_values:
            pred_expanded = expand_neighborhood(pred_shoreline, k)
            gt_expanded = expand_neighborhood(gt_shoreline, k)

            # Compute bidirectional SCR
            scr_pred_to_gt = compute_scr_single(pred_shoreline, gt_expanded)
            scr_gt_to_pred = compute_scr_single(gt_shoreline, pred_expanded)
            scr_k_for_this_image = (scr_pred_to_gt + scr_gt_to_pred) / 2.0

            scr_scores_by_k[k].append(scr_k_for_this_image)

    # Compute mean for each k
    final_results = {}
    for k, scores_list in scr_scores_by_k.items():
        final_results[f'SCR_k{k}_mean'] = np.mean(scores_list)

    # Also compute overall mSCR (average of all k means)
    final_results['mSCR'] = np.mean([final_results[f'SCR_k{k}_mean'] for k in k_values])

    return final_results

# ============================================================================
# DIFFERENTIABLE mSCR LOSS (Same as original)
# ============================================================================

def extract_shoreline_torch(mask):
    """Extract shoreline using differentiable operations"""
    mask_float = mask.float().unsqueeze(1)

    laplacian_kernel = torch.tensor([
        [1, 1, 1],
        [1, -8, 1],
        [1, 1, 1]
    ], dtype=torch.float32, device=mask.device).view(1, 1, 3, 3)

    edges = F.conv2d(mask_float, laplacian_kernel, padding=1)
    shoreline = (edges.abs() > 0.1).float().squeeze(1)

    return shoreline


def expand_neighborhood_torch(shoreline, k):
    """Expand shoreline using max pooling"""
    if k == 4:
        kernel_size = 3
    elif k == 8:
        kernel_size = 3
    elif k == 24:
        kernel_size = 5
    elif k == 48:
        kernel_size = 7
    else:
        kernel_size = 3

    shoreline_expanded = shoreline.unsqueeze(1)
    padding = kernel_size // 2
    expanded = F.max_pool2d(shoreline_expanded, kernel_size=kernel_size,
                           stride=1, padding=padding)

    return expanded.squeeze(1)


def compute_scr_differentiable(pred_mask, gt_mask, k):
    """Compute differentiable SCR for a single k value"""
    batch_size = pred_mask.shape[0]

    pred_shoreline = extract_shoreline_torch(pred_mask)
    gt_shoreline = extract_shoreline_torch(gt_mask)

    pred_expanded = expand_neighborhood_torch(pred_shoreline, k)
    gt_expanded = expand_neighborhood_torch(gt_shoreline, k)

    intersection_1 = (pred_shoreline * gt_expanded).sum(dim=[1, 2])
    card_pred = pred_shoreline.sum(dim=[1, 2])
    card_gt = gt_shoreline.sum(dim=[1, 2])

    intersection_2 = (gt_shoreline * pred_expanded).sum(dim=[1, 2])

    eps = 1e-6
    scr_1 = (2.0 * intersection_1) / (card_pred + card_gt + eps)
    scr_2 = (2.0 * intersection_2) / (card_gt + card_pred + eps)

    scr_bidirectional = (scr_1 + scr_2) / 2.0

    return scr_bidirectional.mean()


class mSCRLoss(nn.Module):
    """Differentiable mSCR Loss"""

    def __init__(self, k_values=[4, 8, 24, 48]):
        super(mSCRLoss, self).__init__()
        self.k_values = k_values

    def forward(self, pred_logits, target):
        pred_probs = F.softmax(pred_logits, dim=1)
        pred_soft = pred_probs[:, 1, :, :]
        target_binary = (target > 0).float()

        scr_scores = []
        for k in self.k_values:
            pred_binary = (pred_soft > 0.5).float()
            scr_k = compute_scr_differentiable(pred_binary, target_binary, k)
            scr_scores.append(scr_k)

        mscr = torch.stack(scr_scores).mean()
        loss = 1.0 - mscr

        return loss, mscr  # Return both loss and mSCR value

# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

class DiceLoss(nn.Module):
    """Dice Loss for binary segmentation"""

    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.softmax(pred, dim=1)
        target_one_hot = F.one_hot(target, num_classes=pred.shape[1]).permute(0, 3, 1, 2).float()

        pred = pred.contiguous().view(-1)
        target_one_hot = target_one_hot.contiguous().view(-1)

        intersection = (pred * target_one_hot).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target_one_hot.sum() + self.smooth)

        return 1 - dice


class CombinedLoss(nn.Module):
    """Combined Loss: CE + Dice + mSCR"""

    def __init__(self, weight_ce=1.0, weight_dice=1.0, weight_mscr=0.5):
        super(CombinedLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss()
        self.mscr_loss = mSCRLoss(k_values=cfg.K_NEIGHBORS)

        self.weight_ce = weight_ce
        self.weight_dice = weight_dice
        self.weight_mscr = weight_mscr

    def forward(self, pred, target):
        ce = self.ce_loss(pred, target)
        dice = self.dice_loss(pred, target)
        mscr_loss, mscr_value = self.mscr_loss(pred, target)

        total_loss = (self.weight_ce * ce +
                     self.weight_dice * dice +
                     self.weight_mscr * mscr_loss)

        return total_loss, (ce.item(), dice.item(), mscr_loss.item(), mscr_value.item())

# ============================================================================
# BOUNDARY-AWARE LOSS
# ============================================================================

class BoundaryAwareLoss(nn.Module):
    """
    Boundary-Aware Loss
    Adds explicit supervision for boundary pixels
    FIXED: Use BCEWithLogitsLoss for autocast compatibility
    """
    def __init__(self):
        super(BoundaryAwareLoss, self).__init__()
        # Use BCEWithLogitsLoss instead of BCELoss (autocast-safe)
        self.bce_loss = nn.BCEWithLogitsLoss()
    
    def forward(self, boundary_pred, target_mask):
        """
        Args:
            boundary_pred: Predicted boundary logits (B, 1, H, W) - NO sigmoid applied
            target_mask: Ground truth segmentation mask (B, H, W)
        """
        # Extract ground truth boundaries
        target_boundary = extract_shoreline_torch(target_mask)
        target_boundary = target_boundary.unsqueeze(1).float()
        
        # Compute BCE loss with logits (applies sigmoid internally)
        loss = self.bce_loss(boundary_pred, target_boundary)
        
        return loss


class EnhancedCombinedLoss(nn.Module):
    """
    Enhanced Combined Loss: CE + Dice + mSCR + Boundary
    """

    def __init__(self, weight_ce=1.0, weight_dice=1.0, weight_mscr=0.5, weight_boundary=0.3):
        super(EnhancedCombinedLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss()
        self.mscr_loss = mSCRLoss(k_values=cfg.K_NEIGHBORS)
        self.boundary_loss = BoundaryAwareLoss()

        self.weight_ce = weight_ce
        self.weight_dice = weight_dice
        self.weight_mscr = weight_mscr
        self.weight_boundary = weight_boundary

    def forward(self, pred, boundary_pred, target):
        ce = self.ce_loss(pred, target)
        dice = self.dice_loss(pred, target)
        mscr_loss, mscr_value = self.mscr_loss(pred, target)
        boundary = self.boundary_loss(boundary_pred, target)

        total_loss = (self.weight_ce * ce +
                     self.weight_dice * dice +
                     self.weight_mscr * mscr_loss +
                     self.weight_boundary * boundary)

        return total_loss, (ce.item(), dice.item(), mscr_loss.item(), mscr_value.item(), boundary.item())

# ============================================================================
# METRICS
# ============================================================================

def compute_iou(pred, target, num_classes=2):
    """Compute Intersection over Union for each class"""
    ious = []
    pred = pred.view(-1)
    target = target.view(-1)

    for cls in range(num_classes):
        pred_cls = (pred == cls)
        target_cls = (target == cls)

        intersection = (pred_cls & target_cls).sum().float()
        union = (pred_cls | target_cls).sum().float()

        if union == 0:
            iou = float('nan')
        else:
            iou = (intersection / union).item()

        ious.append(iou)

    return ious


def compute_accuracy(pred, target):
    """Compute pixel-wise accuracy"""
    correct = (pred == target).sum().item()
    total = target.numel()
    return correct / total

# ============================================================================
# TRAINING & VALIDATION FUNCTIONS
# ============================================================================

def train_one_epoch(model, dataloader, criterion, optimizer, scaler, device, epoch):
    """Train for one epoch with boundary supervision"""

    model.train()

    # Metric accumulators
    running_loss = 0.0
    running_ce_loss = 0.0
    running_dice_loss = 0.0
    running_mscr_loss = 0.0
    running_mscr_diff = 0.0
    running_boundary_loss = 0.0  # New
    running_acc = 0.0
    running_iou_land = 0.0
    running_iou_water = 0.0
    running_mscr_eval = 0.0
    running_scr_k4 = 0.0
    running_scr_k8 = 0.0
    running_scr_k24 = 0.0
    running_scr_k48 = 0.0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} [TRAIN]")

    for batch_idx, (images, masks) in enumerate(pbar):
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        # Mixed precision training
        if cfg.USE_MIXED_PRECISION:
            with autocast():
                seg_outputs, boundary_outputs = model(images)  # Now returns 2 outputs
                loss, (ce_loss, dice_loss, mscr_loss, mscr_diff, boundary_loss) = criterion(
                    seg_outputs, boundary_outputs, masks
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            seg_outputs, boundary_outputs = model(images)
            loss, (ce_loss, dice_loss, mscr_loss, mscr_diff, boundary_loss) = criterion(
                seg_outputs, boundary_outputs, masks
            )
            loss.backward()
            optimizer.step()

        # Compute metrics
        pred = torch.argmax(seg_outputs, dim=1)

        acc = compute_accuracy(pred, masks)
        ious = compute_iou(pred, masks, num_classes=cfg.NUM_CLASSES)

        # Compute non-differentiable mSCR (evaluation version)
        mscr_scores = compute_scr_by_k_for_batch(pred, masks, k_values=cfg.K_NEIGHBORS)

        # Update running metrics
        running_loss += loss.item()
        running_ce_loss += ce_loss
        running_dice_loss += dice_loss
        running_mscr_loss += mscr_loss
        running_mscr_diff += mscr_diff
        running_boundary_loss += boundary_loss
        running_acc += acc
        running_iou_land += ious[0] if not np.isnan(ious[0]) else 0
        running_iou_water += ious[1] if not np.isnan(ious[1]) else 0
        running_mscr_eval += mscr_scores['mSCR']
        running_scr_k4 += mscr_scores['SCR_k4_mean']
        running_scr_k8 += mscr_scores['SCR_k8_mean']
        running_scr_k24 += mscr_scores['SCR_k24_mean']
        running_scr_k48 += mscr_scores['SCR_k48_mean']

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'mSCR_e': f'{mscr_scores["mSCR"]:.4f}',
            'bound': f'{boundary_loss:.4f}'
        })

        # Log every 50 batches
        if batch_idx % 50 == 0:
            logger.debug(f"Epoch {epoch+1} Batch {batch_idx}/{len(dataloader)} - "
                        f"Loss: {loss.item():.4f}, Boundary: {boundary_loss:.4f}, "
                        f"mSCR_eval: {mscr_scores['mSCR']:.4f}")

    # Average metrics
    num_batches = len(dataloader)
    metrics = {
        'loss': running_loss / num_batches,
        'ce_loss': running_ce_loss / num_batches,
        'dice_loss': running_dice_loss / num_batches,
        'mscr_loss': running_mscr_loss / num_batches,
        'mscr_diff': running_mscr_diff / num_batches,
        'boundary_loss': running_boundary_loss / num_batches,
        'acc': running_acc / num_batches,
        'iou': (running_iou_land + running_iou_water) / (2 * num_batches),
        'iou_land': running_iou_land / num_batches,
        'iou_water': running_iou_water / num_batches,
        'mscr_eval': running_mscr_eval / num_batches,
        'scr_k4': running_scr_k4 / num_batches,
        'scr_k8': running_scr_k8 / num_batches,
        'scr_k24': running_scr_k24 / num_batches,
        'scr_k48': running_scr_k48 / num_batches
    }

    return metrics


def validate(model, dataloader, criterion, device, epoch):
    """Validate the model with boundary supervision"""

    model.eval()

    # Metric accumulators
    running_loss = 0.0
    running_ce_loss = 0.0
    running_dice_loss = 0.0
    running_mscr_loss = 0.0
    running_mscr_diff = 0.0
    running_boundary_loss = 0.0  # New
    running_acc = 0.0
    running_iou_land = 0.0
    running_iou_water = 0.0
    running_mscr_eval = 0.0
    running_scr_k4 = 0.0
    running_scr_k8 = 0.0
    running_scr_k24 = 0.0
    running_scr_k48 = 0.0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} [VAL]  ")

    with torch.no_grad():
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)

            seg_outputs, boundary_outputs = model(images)
            loss, (ce_loss, dice_loss, mscr_loss, mscr_diff, boundary_loss) = criterion(
                seg_outputs, boundary_outputs, masks
            )

            # Compute metrics
            pred = torch.argmax(seg_outputs, dim=1)

            acc = compute_accuracy(pred, masks)
            ious = compute_iou(pred, masks, num_classes=cfg.NUM_CLASSES)

            # Compute non-differentiable mSCR
            mscr_scores = compute_scr_by_k_for_batch(pred, masks, k_values=cfg.K_NEIGHBORS)

            # Update running metrics
            running_loss += loss.item()
            running_ce_loss += ce_loss
            running_dice_loss += dice_loss
            running_mscr_loss += mscr_loss
            running_mscr_diff += mscr_diff
            running_boundary_loss += boundary_loss
            running_acc += acc
            running_iou_land += ious[0] if not np.isnan(ious[0]) else 0
            running_iou_water += ious[1] if not np.isnan(ious[1]) else 0
            running_mscr_eval += mscr_scores['mSCR']
            running_scr_k4 += mscr_scores['SCR_k4_mean']
            running_scr_k8 += mscr_scores['SCR_k8_mean']
            running_scr_k24 += mscr_scores['SCR_k24_mean']
            running_scr_k48 += mscr_scores['SCR_k48_mean']

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'mSCR_e': f'{mscr_scores["mSCR"]:.4f}',
                'bound': f'{boundary_loss:.4f}'
            })

    # Average metrics
    num_batches = len(dataloader)
    metrics = {
        'loss': running_loss / num_batches,
        'ce_loss': running_ce_loss / num_batches,
        'dice_loss': running_dice_loss / num_batches,
        'mscr_loss': running_mscr_loss / num_batches,
        'mscr_diff': running_mscr_diff / num_batches,
        'boundary_loss': running_boundary_loss / num_batches,
        'acc': running_acc / num_batches,
        'iou': (running_iou_land + running_iou_water) / (2 * num_batches),
        'iou_land': running_iou_land / num_batches,
        'iou_water': running_iou_water / num_batches,
        'mscr_eval': running_mscr_eval / num_batches,
        'scr_k4': running_scr_k4 / num_batches,
        'scr_k8': running_scr_k8 / num_batches,
        'scr_k24': running_scr_k24 / num_batches,
        'scr_k48': running_scr_k48 / num_batches
    }

    return metrics

# ============================================================================
# VISUALIZATION
# ============================================================================

class MetricsPlotter:
    """Enhanced metrics plotting with per-k tracking"""

    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.train_ce_losses = []
        self.val_ce_losses = []
        self.train_dice_losses = []
        self.val_dice_losses = []
        self.train_mscr_losses = []
        self.val_mscr_losses = []

        self.train_accs = []
        self.val_accs = []
        self.train_ious = []
        self.val_ious = []

        # Differentiable mSCR
        self.train_mscr_diff = []
        self.val_mscr_diff = []

        # Evaluation mSCR
        self.train_mscr_eval = []
        self.val_mscr_eval = []

        # Per-k SCR scores
        self.train_scr_k4s = []
        self.val_scr_k4s = []
        self.train_scr_k8s = []
        self.val_scr_k8s = []
        self.train_scr_k24s = []
        self.val_scr_k24s = []
        self.train_scr_k48s = []
        self.val_scr_k48s = []

    def update(self, train_metrics, val_metrics):
        """Update plots with new metrics"""

        # Loss components
        self.train_losses.append(train_metrics['loss'])
        self.val_losses.append(val_metrics['loss'])
        self.train_ce_losses.append(train_metrics['ce_loss'])
        self.val_ce_losses.append(val_metrics['ce_loss'])
        self.train_dice_losses.append(train_metrics['dice_loss'])
        self.val_dice_losses.append(val_metrics['dice_loss'])
        self.train_mscr_losses.append(train_metrics['mscr_loss'])
        self.val_mscr_losses.append(val_metrics['mscr_loss'])

        # Performance metrics
        self.train_accs.append(train_metrics['acc'])
        self.val_accs.append(val_metrics['acc'])
        self.train_ious.append(train_metrics['iou'])
        self.val_ious.append(val_metrics['iou'])

        # mSCR values
        self.train_mscr_diff.append(train_metrics['mscr_diff'])
        self.val_mscr_diff.append(val_metrics['mscr_diff'])
        self.train_mscr_eval.append(train_metrics['mscr_eval'])
        self.val_mscr_eval.append(val_metrics['mscr_eval'])

        # Per-k scores
        self.train_scr_k4s.append(train_metrics['scr_k4'])
        self.val_scr_k4s.append(val_metrics['scr_k4'])
        self.train_scr_k8s.append(train_metrics['scr_k8'])
        self.val_scr_k8s.append(val_metrics['scr_k8'])
        self.train_scr_k24s.append(train_metrics['scr_k24'])
        self.val_scr_k24s.append(val_metrics['scr_k24'])
        self.train_scr_k48s.append(train_metrics['scr_k48'])
        self.val_scr_k48s.append(val_metrics['scr_k48'])

        self._create_and_display_plot()

    def _create_and_display_plot(self):
        """Create comprehensive plot"""

        plt.close('all')

        fig = plt.figure(figsize=(24, 20))
        gs = fig.add_gridspec(4, 2, hspace=0.35, wspace=0.25)

        epochs = range(1, len(self.train_losses) + 1)

        # Plot 1: Total Loss
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(epochs, self.train_losses, 'b-o', label='Train Loss', linewidth=2, markersize=5)
        ax1.plot(epochs, self.val_losses, 'r-s', label='Val Loss', linewidth=2, markersize=5)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Total Loss (CE + Dice + mSCR)', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)

        # Plot 2: Loss Components
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(epochs, self.val_ce_losses, 'g-^', label='Val CE', linewidth=2, markersize=4)
        ax2.plot(epochs, self.val_dice_losses, 'c-v', label='Val Dice', linewidth=2, markersize=4)
        ax2.plot(epochs, self.val_mscr_losses, 'm-d', label='Val mSCR Loss', linewidth=2, markersize=4)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Loss Value', fontsize=12)
        ax2.set_title('Validation Loss Components', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)

        # Plot 3: Accuracy
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot(epochs, self.train_accs, 'b-o', label='Train Acc', linewidth=2, markersize=5)
        ax3.plot(epochs, self.val_accs, 'r-s', label='Val Acc', linewidth=2, markersize=5)
        ax3.set_xlabel('Epoch', fontsize=12)
        ax3.set_ylabel('Accuracy', fontsize=12)
        ax3.set_title('Pixel-wise Accuracy', fontsize=14, fontweight='bold')
        ax3.legend(fontsize=11)
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim([0, 1])

        # Plot 4: IoU
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.plot(epochs, self.train_ious, 'b-o', label='Train IoU', linewidth=2, markersize=5)
        ax4.plot(epochs, self.val_ious, 'r-s', label='Val IoU', linewidth=2, markersize=5)
        ax4.set_xlabel('Epoch', fontsize=12)
        ax4.set_ylabel('Mean IoU', fontsize=12)
        ax4.set_title('Intersection over Union', fontsize=14, fontweight='bold')
        ax4.legend(fontsize=11)
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim([0, 1])

        # Plot 5: mSCR Comparison (Differentiable vs Evaluation)
        ax5 = fig.add_subplot(gs[2, 0])
        ax5.plot(epochs, self.val_mscr_diff, 'b-o', label='Val mSCR (Differentiable)', linewidth=2.5, markersize=6)
        ax5.plot(epochs, self.val_mscr_eval, 'r-s', label='Val mSCR (Evaluation)', linewidth=2.5, markersize=6)
        ax5.set_xlabel('Epoch', fontsize=12)
        ax5.set_ylabel('mSCR Score', fontsize=12)
        ax5.set_title('mSCR: Differentiable vs Evaluation', fontsize=14, fontweight='bold')
        ax5.legend(fontsize=11)
        ax5.grid(True, alpha=0.3)
        ax5.set_ylim([0, 1])

        # Plot 6: Per-k SCR Scores
        ax6 = fig.add_subplot(gs[2, 1])
        ax6.plot(epochs, self.val_mscr_eval, 'k-', label='Val mSCR (Mean)', linewidth=3, markersize=6, alpha=0.7)
        ax6.plot(epochs, self.val_scr_k4s, 'g--o', label='Val SCR k=4', linewidth=2, markersize=4, alpha=0.8)
        ax6.plot(epochs, self.val_scr_k8s, 'c--s', label='Val SCR k=8', linewidth=2, markersize=4, alpha=0.8)
        ax6.plot(epochs, self.val_scr_k24s, 'm--^', label='Val SCR k=24', linewidth=2, markersize=4, alpha=0.8)
        ax6.plot(epochs, self.val_scr_k48s, 'y--v', label='Val SCR k=48', linewidth=2, markersize=4, alpha=0.8)
        ax6.set_xlabel('Epoch', fontsize=12)
        ax6.set_ylabel('SCR Score', fontsize=12)
        ax6.set_title('SCR by Neighborhood Size (k)', fontsize=14, fontweight='bold')
        ax6.legend(fontsize=10)
        ax6.grid(True, alpha=0.3)
        ax6.set_ylim([0, 1])

        # Plot 7: Train vs Val mSCR (Evaluation)
        ax7 = fig.add_subplot(gs[3, 0])
        ax7.plot(epochs, self.train_mscr_eval, 'b-o', label='Train mSCR', linewidth=2.5, markersize=6, alpha=0.7)
        ax7.plot(epochs, self.val_mscr_eval, 'r-s', label='Val mSCR', linewidth=2.5, markersize=6)
        ax7.set_xlabel('Epoch', fontsize=12)
        ax7.set_ylabel('mSCR Score', fontsize=12)
        ax7.set_title('mSCR Evaluation Metric', fontsize=14, fontweight='bold')
        ax7.legend(fontsize=11)
        ax7.grid(True, alpha=0.3)
        ax7.set_ylim([0, 1])

        # Plot 8: Per-k Comparison (Train vs Val for k=4)
        ax8 = fig.add_subplot(gs[3, 1])
        ax8.plot(epochs, self.train_scr_k4s, 'b:', label='Train k=4', linewidth=2, alpha=0.6)
        ax8.plot(epochs, self.val_scr_k4s, 'g-o', label='Val k=4', linewidth=2.5, markersize=5)
        ax8.plot(epochs, self.train_scr_k48s, 'c:', label='Train k=48', linewidth=2, alpha=0.6)
        ax8.plot(epochs, self.val_scr_k48s, 'y-s', label='Val k=48', linewidth=2.5, markersize=5)
        ax8.set_xlabel('Epoch', fontsize=12)
        ax8.set_ylabel('SCR Score', fontsize=12)
        ax8.set_title('SCR k=4 vs k=48 (Strict vs Lenient)', fontsize=14, fontweight='bold')
        ax8.legend(fontsize=10)
        ax8.grid(True, alpha=0.3)
        ax8.set_ylim([0, 1])

        # Save and display
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)

        clear_output(wait=True)
        display(IPImage(buf.read()))

        plt.close(fig)

    def save_final_plot(self, save_path):
        """Save final training curves"""
        self._create_and_display_plot()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Final training curves saved to: {save_path}")

# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================

def main():
    """Main training function"""

    logger.info("=" * 80)
    logger.info("Enhanced HRNet-W48 with Attention + Boundary Supervision")
    logger.info("=" * 80)
    logger.info(f"Device: {cfg.DEVICE}")
    logger.info(f"Batch Size: {cfg.BATCH_SIZE}")
    logger.info(f"Number of Epochs: {cfg.NUM_EPOCHS}")
    logger.info(f"Learning Rate: {cfg.LEARNING_RATE}")
    logger.info(f"Mixed Precision: {cfg.USE_MIXED_PRECISION}")
    logger.info(f"Input Channels: {cfg.INPUT_CHANNELS} (RGB + Sobel + Canny)")
    logger.info(f"K-Neighbors for mSCR: {cfg.K_NEIGHBORS}")
    logger.info(f"Loss Weights - CE: {cfg.LOSS_WEIGHT_CE}, Dice: {cfg.LOSS_WEIGHT_DICE}, mSCR: {cfg.LOSS_WEIGHT_MSCR}")
    logger.info("=" * 80)

    # Create datasets
    logger.info("\nLoading datasets...")
    train_dataset = CAIDHybridDataset(
        split_file=cfg.TRAIN_TXT,
        images_dir=cfg.IMAGES_DIR,
        masks_dir=cfg.MASKS_DIR,
        augment=True
    )

    val_dataset = CAIDHybridDataset(
        split_file=cfg.VAL_TXT,
        images_dir=cfg.IMAGES_DIR,
        masks_dir=cfg.MASKS_DIR,
        augment=False
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY
    )

    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Val batches: {len(val_loader)}")

    # Create model
    logger.info("\nInitializing Enhanced HRNet-W48 with attention mechanisms...")
    model = HRNetW48(
        num_classes=cfg.NUM_CLASSES,
        input_channels=cfg.INPUT_CHANNELS
    ).to(cfg.DEVICE)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    # Enhanced loss function with boundary supervision
    criterion = EnhancedCombinedLoss(
        weight_ce=cfg.LOSS_WEIGHT_CE,
        weight_dice=cfg.LOSS_WEIGHT_DICE,
        weight_mscr=cfg.LOSS_WEIGHT_MSCR,
        weight_boundary=0.3  # New boundary loss weight
    )
    
    # Optimizer and scheduler
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.LEARNING_RATE,
        weight_decay=cfg.WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    scaler = GradScaler() if cfg.USE_MIXED_PRECISION else None

    plotter = MetricsPlotter()
    history = []
    best_val_mscr = 0.0

    logger.info("\n" + "=" * 80)
    logger.info("Starting Training with Attention + Boundary Supervision...")
    logger.info("=" * 80)

    # Training loop
    for epoch in range(cfg.NUM_EPOCHS):
        start_time = time.time()

        # Train
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, cfg.DEVICE, epoch
        )

        # Validate
        val_metrics = validate(
            model, val_loader, criterion, cfg.DEVICE, epoch
        )

        epoch_time = time.time() - start_time

        # Update scheduler
        scheduler.step(val_metrics['loss'])

        # Log epoch summary (include boundary loss)
        logger.info("\n" + "=" * 80)
        logger.info(f"EPOCH {epoch + 1}/{cfg.NUM_EPOCHS} Summary (Time: {epoch_time:.2f}s)")
        logger.info("=" * 80)
        logger.info(f"LOSSES:")
        logger.info(f"  Total     - Train: {train_metrics['loss']:.4f} | Val: {val_metrics['loss']:.4f}")
        logger.info(f"  CE        - Train: {train_metrics['ce_loss']:.4f} | Val: {val_metrics['ce_loss']:.4f}")
        logger.info(f"  Dice      - Train: {train_metrics['dice_loss']:.4f} | Val: {val_metrics['dice_loss']:.4f}")
        logger.info(f"  mSCR Loss - Train: {train_metrics['mscr_loss']:.4f} | Val: {val_metrics['mscr_loss']:.4f}")
        logger.info(f"  Boundary  - Train: {train_metrics['boundary_loss']:.4f} | Val: {val_metrics['boundary_loss']:.4f}")
        logger.info(f"\nMETRICS:")
        logger.info(f"  Accuracy  - Train: {train_metrics['acc']:.4f} | Val: {val_metrics['acc']:.4f}")
        logger.info(f"  IoU       - Train: {train_metrics['iou']:.4f} | Val: {val_metrics['iou']:.4f}")
        logger.info(f"\nmSCR SCORES:")
        logger.info(f"  Evaluation - Train: {train_metrics['mscr_eval']:.4f} | Val: {val_metrics['mscr_eval']:.4f}")
        logger.info(f"\nSCR by K-values (Validation):")
        logger.info(f"  k=4:  {val_metrics['scr_k4']:.4f}")
        logger.info(f"  k=8:  {val_metrics['scr_k8']:.4f}")
        logger.info(f"  k=24: {val_metrics['scr_k24']:.4f}")
        logger.info(f"  k=48: {val_metrics['scr_k48']:.4f}")

        # Save to history
        history_entry = {
            'epoch': epoch + 1,
            'train_loss': train_metrics['loss'],
            'val_loss': val_metrics['loss'],
            'train_boundary_loss': train_metrics['boundary_loss'],
            'val_boundary_loss': val_metrics['boundary_loss'],
            'train_ce_loss': train_metrics['ce_loss'],
            'val_ce_loss': val_metrics['ce_loss'],
            'train_dice_loss': train_metrics['dice_loss'],
            'val_dice_loss': val_metrics['dice_loss'],
            'train_mscr_loss': train_metrics['mscr_loss'],
            'val_mscr_loss': val_metrics['mscr_loss'],
            'train_acc': train_metrics['acc'],
            'val_acc': val_metrics['acc'],
            'train_iou': train_metrics['iou'],
            'val_iou': val_metrics['iou'],
            'train_mscr_diff': train_metrics['mscr_diff'],
            'val_mscr_diff': val_metrics['mscr_diff'],
            'train_mscr_eval': train_metrics['mscr_eval'],
            'val_mscr_eval': val_metrics['mscr_eval'],
            'train_scr_k4': train_metrics['scr_k4'],
            'val_scr_k4': val_metrics['scr_k4'],
            'train_scr_k8': train_metrics['scr_k8'],
            'val_scr_k8': val_metrics['scr_k8'],
            'train_scr_k24': train_metrics['scr_k24'],
            'val_scr_k24': val_metrics['scr_k24'],
            'train_scr_k48': train_metrics['scr_k48'],
            'val_scr_k48': val_metrics['scr_k48']
        }
        history.append(history_entry)

        # Update plot
        plotter.update(train_metrics, val_metrics)

        # Check for best model based on validation mSCR (evaluation)
        # NOTE: metrics dictionary uses the key 'mscr_eval' (lowercase) —
        # use that consistently to avoid KeyError
        if val_metrics['mscr_eval'] > best_val_mscr:
            best_val_mscr = val_metrics['mscr_eval']
            logger.info(f"New best validation mSCR ({best_val_mscr:.4f}). Saving model...")
            torch.save(model.state_dict(), cfg.BEST_MODEL_PATH)

    logger.info("\n" + "=" * 80)
    logger.info("Training Finished!")
    logger.info("=" * 80)
    logger.info(f"Best Validation mSCR: {best_val_mscr:.4f}")
    logger.info(f"Best model saved to: {cfg.BEST_MODEL_PATH}")

    # Save final plot and history
    plotter.save_final_plot(os.path.join(cfg.OUTPUT_DIR, 'training_curves.png'))
    history_df = pd.DataFrame(history)
    history_df.to_csv(os.path.join(cfg.OUTPUT_DIR, 'training_history.csv'), index=False)
    logger.info(f"Training history saved to: {os.path.join(cfg.OUTPUT_DIR, 'training_history.csv')}")


if __name__ == '__main__':
    main()