"""
HRNet-W48 with Hybrid Edge Channels Training Script for CAID Dataset
Semantic Segmentation: Land vs Water Classification

### FINAL VERSION (v2) - UPGRADED LOSS & OPTIMIZER ###
- BUG FIX: Corrected Sobel Magnitude calculation (sobelx**2)
- METRIC FIX: Reverted SCR calculation to original formula for fair comparison
- LOSS UPGRADE 1: Replaced Cross-Entropy with Focal Loss
- LOSS UPGRADE 2: Replaced Dice Loss with Lovasz-Softmax Loss
- OPTIMIZER UPGRADE: Replaced Adam with AdamW
- SCHEDULER UPGRADE: Replaced ReduceLROnPlateau with CosineAnnealingLR
- TWEAK: Lowered post-processing threshold to 25
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

    # Avoid duplicate handlers if script is re-run in a notebook
    if not logger.hasHandlers():
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
    BATCH_SIZE = 8   # Reduced for HRNet-W48
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-4 # AdamW works well with 1e-4 or 3e-4
    WEIGHT_DECAY = 1e-2 # True weight decay for AdamW

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

    # ### MODIFICATION: New Loss Weights ###
    # We replaced CE/Dice with Focal/Lovasz
    LOSS_WEIGHT_FOCAL = 1.0   # Replaces CE
    LOSS_WEIGHT_LOVASZ = 1.0  # Replaces Dice
    LOSS_WEIGHT_MSCR = 0.5    # Differentiable mSCR loss weight
    LOSS_WEIGHT_BOUNDARY = 0.3 # Auxiliary boundary head weight

    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # ### MODIFICATION: Lowered post-processing threshold ###
    USE_POST_PROCESSING = True
    MIN_AREA_THRESHOLD = 25  # Lowered: Removes only tiny speckles
    
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

    # ### MODIFICATION: BUG FIX ###
    # Corrected formula from sobelx*2 to sobelx**2
    magnitude = np.sqrt(sobelx**2 + sobely**2)

    # Normalize to [0, 1]
    if magnitude.max() > 0:
        magnitude = magnitude / magnitude.max()

    return magnitude.astype(np.float32)


def compute_canny_edges(image_np):
    """
    Compute Canny edges from RGB image.
    """
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, cfg.CANNY_THRESHOLD1, cfg.CANNY_THRESHOLD2)
    edges = edges.astype(np.float32) / 255.0
    return edges


def create_hybrid_input(image):
    """
    Create 5-channel hybrid input: [R, G, B, Sobel, Canny]
    """
    image_np = np.array(image)
    sobel = compute_sobel_magnitude(image_np)
    canny = compute_canny_edges(image_np)

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
    """

    def __init__(self, split_file, images_dir, masks_dir, transform=None, augment=False):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.augment = augment
        with open(split_file, 'r') as f:
            self.image_ids = [line.strip() for line in f.readlines()]
        logger.info(f"Loaded {len(self.image_ids)} samples from {split_file}")

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_path = os.path.join(self.images_dir, f"{img_id}.png")
        mask_path = os.path.join(self.masks_dir, f"{img_id}.png")

        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L') 

        if self.augment:
            image, mask = self.apply_augmentation(image, mask)

        if cfg.USE_EDGE_CHANNELS:
            hybrid_input = create_hybrid_input(image)  # (H, W, 5)
            rgb = hybrid_input[:, :, :3] / 255.0  # [0, 1]
            edges = hybrid_input[:, :, 3:]      # Already [0, 1]
            
            rgb_normalized = (rgb - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
            edges_normalized = edges * 2.0 - 1.0 # Normalize edges to [-1, 1]
            final_input = np.concatenate([rgb_normalized, edges_normalized], axis=2)
            
            image_tensor = torch.from_numpy(final_input).permute(2, 0, 1).float()
        else:
            transform = T.Compose([
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            image_tensor = transform(image)

        mask = torch.from_numpy(np.array(mask)).long()
        return image_tensor, mask

    def apply_augmentation(self, image, mask):
        if np.random.rand() < cfg.FLIP_PROB:
            image = T.functional.hflip(image)
            mask = T.functional.hflip(mask)
        if np.random.rand() < cfg.FLIP_PROB:
            image = T.functional.vflip(image)
            mask = T.functional.vflip(mask)
        if np.random.rand() < cfg.ROTATE_PROB:
            angle = int(np.random.choice([90, 180, 270]))
            image = T.functional.rotate(image, angle)
            mask = T.functional.rotate(mask, angle)
        return image, mask

# ============================================================================
# HRNET-W48 MODEL IMPLEMENTATION
# ============================================================================

class BasicBlock(nn.Module):
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
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)
    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x

class BoundaryRefinementModule(nn.Module):
    def __init__(self, in_channels):
        super(BoundaryRefinementModule, self).__init__()
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
    def __init__(self, channels_list=[48, 96, 192, 384], out_channels=256):
        super(EnhancedMultiScaleFusion, self).__init__()
        self.scale_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, out_channels // 4, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels // 4),
                nn.ReLU(inplace=True),
                CBAM(out_channels // 4, reduction=8)
            ) for ch in channels_list
        ])
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            CBAM(out_channels, reduction=16)
        )
        self.boundary_refine = BoundaryRefinementModule(out_channels)
    
    def forward(self, features_list, target_size):
        processed_features = []
        for i, (feat, conv) in enumerate(zip(features_list, self.scale_convs)):
            feat_up = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)
            feat_processed = conv(feat_up)
            processed_features.append(feat_processed)
        
        fused = torch.cat(processed_features, dim=1)
        fused = self.fusion_conv(fused)
        refined = self.boundary_refine(fused)
        return refined

# ============================================================================
# ENHANCED HRNET-W48 MODEL WITH ATTENTION
# ============================================================================

class HRNetW48(nn.Module):
    def __init__(self, num_classes=2, input_channels=5):
        super(HRNetW48, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 64, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 3, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.input_attention = ChannelAttention(64, reduction=8)
        self.layer1 = self._make_layer(BasicBlock, 64, 64, 4)

        # Simplified transitions & stages for HRNet-W48 structure
        self.trans1_branch1 = nn.Sequential(nn.Conv2d(64, 48, 3, 1, 1, bias=False), nn.BatchNorm2d(48), nn.ReLU(inplace=True))
        self.trans1_branch2 = nn.Sequential(nn.Conv2d(64, 96, 3, 2, 1, bias=False), nn.BatchNorm2d(96), nn.ReLU(inplace=True))
        self.stage2_branch1 = self._make_layer(BasicBlock, 48, 48, 4)
        self.stage2_branch2 = self._make_layer(BasicBlock, 96, 96, 4)
        self.trans2_branch3 = nn.Sequential(nn.Conv2d(96, 192, 3, 2, 1, bias=False), nn.BatchNorm2d(192), nn.ReLU(inplace=True))
        self.stage3_branch1 = self._make_layer(BasicBlock, 48, 48, 4)
        self.stage3_branch2 = self._make_layer(BasicBlock, 96, 96, 4)
        self.stage3_branch3 = self._make_layer(BasicBlock, 192, 192, 4)
        self.trans3_branch4 = nn.Sequential(nn.Conv2d(192, 384, 3, 2, 1, bias=False), nn.BatchNorm2d(384), nn.ReLU(inplace=True))
        self.stage4_branch1 = self._make_layer(BasicBlock, 48, 48, 4)
        self.stage4_branch2 = self._make_layer(BasicBlock, 96, 96, 4)
        self.stage4_branch3 = self._make_layer(BasicBlock, 192, 192, 4)
        self.stage4_branch4 = self._make_layer(BasicBlock, 384, 384, 4)
        
        self.stage4_attention = nn.ModuleList([
            CBAM(48, reduction=8), CBAM(96, reduction=8),
            CBAM(192, reduction=8), CBAM(384, reduction=8)
        ])
        self.multi_scale_fusion = EnhancedMultiScaleFusion(
            channels_list=[48, 96, 192, 384], out_channels=256
        )
        
        self.seg_head = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Dropout2d(0.1), nn.Conv2d(128, num_classes, 1)
        )
        
        # ### MODIFICATION: Added Dropout to boundary_head ###
        self.boundary_head = nn.Sequential(
            nn.Conv2d(256, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Dropout2d(0.1), # Added dropout for regularization
            nn.Conv2d(64, 1, 1) 
        )
        self._init_weights()

    def _make_layer(self, block, inplanes, planes, blocks):
        layers = []
        layers.append(block(inplanes, planes))
        for _ in range(1, blocks):
            layers.append(block(planes, planes))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        input_size = x.size()[2:]
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.input_attention(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.layer1(x)

        # Simplified forward pass
        x_s2_b1 = self.stage2_branch1(self.trans1_branch1(x))
        x_s2_b2 = self.stage2_branch2(self.trans1_branch2(x))
        x_s3_b1 = self.stage3_branch1(x_s2_b1)
        x_s3_b2 = self.stage3_branch2(x_s2_b2)
        x_s3_b3 = self.stage3_branch3(self.trans2_branch3(x_s2_b2))
        x_s4_b1 = self.stage4_branch1(x_s3_b1)
        x_s4_b2 = self.stage4_branch2(x_s3_b2)
        x_s4_b3 = self.stage4_branch3(x_s3_b3)
        x_s4_b4 = self.stage4_branch4(self.trans3_branch4(x_s3_b3))

        y_list = [
            self.stage4_attention[0](x_s4_b1), self.stage4_attention[1](x_s4_b2),
            self.stage4_attention[2](x_s4_b3), self.stage4_attention[3](x_s4_b4)
        ]

        target_size = (y_list[0].size(2), y_list[0].size(3))
        fused_features = self.multi_scale_fusion(y_list, target_size)
        seg_out = self.seg_head(fused_features)
        boundary_out = self.boundary_head(fused_features)
        
        seg_out = F.interpolate(seg_out, size=input_size, mode='bilinear', align_corners=False)
        boundary_out = F.interpolate(boundary_out, size=input_size, mode='bilinear', align_corners=False)

        return seg_out, boundary_out

# ============================================================================
# MODIFIED mSCR METRIC EVALUATION (Per-k Averaging)
# ============================================================================

def extract_shoreline_pixels(mask):
    """Extract shoreline (boundary) pixels from a binary mask"""
    binary_mask = (mask > 0).astype(np.uint8)
    # Use the original (scipy) implementation for boundary extraction
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
    
    # ### MODIFICATION: METRIC FIX ###
    # Reverting to the original paper's formula for apples-to-apples comparison
    scr = (2.0 * intersection) / (S1 + S2_expanded) 
    
    return scr


def compute_scr_by_k_for_batch(pred_batch, target_batch, k_values=[4, 8, 24, 48]):
    """
    Computes SCR for a batch, averaging across IMAGES for EACH k-value.
    """
    if torch.is_tensor(pred_batch):
        pred_batch = pred_batch.cpu().numpy()
    if torch.is_tensor(target_batch):
        target_batch = target_batch.cpu().numpy()
    batch_size = pred_batch.shape[0]
    scr_scores_by_k = {k: [] for k in k_values}

    for i in range(batch_size):
        pred_shoreline = extract_shoreline_pixels(pred_batch[i])
        gt_shoreline = extract_shoreline_pixels(target_batch[i])

        for k in k_values:
            pred_expanded = expand_neighborhood(pred_shoreline, k)
            gt_expanded = expand_neighborhood(gt_shoreline, k)
            
            scr_pred_to_gt = compute_scr_single(pred_shoreline, gt_expanded)
            scr_gt_to_pred = compute_scr_single(gt_shoreline, pred_expanded)
            scr_k_for_this_image = (scr_pred_to_gt + scr_gt_to_pred) / 2.0
            scr_scores_by_k[k].append(scr_k_for_this_image)

    final_results = {}
    for k, scores_list in scr_scores_by_k.items():
        final_results[f'SCR_k{k}_mean'] = np.mean(scores_list)
    final_results['mSCR'] = np.mean([final_results[f'SCR_k{k}_mean'] for k in k_values])
    return final_results

# ============================================================================
# DIFFERENTIABLE mSCR LOSS (Unchanged)
# ============================================================================

def extract_shoreline_torch(mask):
    if mask.dim() == 4:
        mask = mask.squeeze(1)
    mask_float = mask.float().unsqueeze(1) # B, 1, H, W
    laplacian_kernel = torch.tensor([
        [1, 1, 1], [1, -8, 1], [1, 1, 1]
    ], dtype=torch.float32, device=mask.device).view(1, 1, 3, 3)
    edges = F.conv2d(mask_float, laplacian_kernel, padding=1)
    shoreline = (edges.abs() > 0.1).float().squeeze(1) # B, H, W
    return shoreline

def expand_neighborhood_torch(shoreline, k):
    if k == 4: kernel_size = 3
    elif k == 8: kernel_size = 3
    elif k == 24: kernel_size = 5
    elif k == 48: kernel_size = 7
    else: kernel_size = 3
    shoreline_expanded = shoreline.unsqueeze(1) # B, 1, H, W
    padding = kernel_size // 2
    expanded = F.max_pool2d(shoreline_expanded, kernel_size=kernel_size,
                            stride=1, padding=padding)
    return expanded.squeeze(1) # B, H, W

def compute_scr_differentiable(pred_mask, gt_mask, k):
    """Compute differentiable SCR for a single k value"""
    pred_shoreline = extract_shoreline_torch(pred_mask)
    gt_shoreline = extract_shoreline_torch(gt_mask)
    pred_expanded = expand_neighborhood_torch(pred_shoreline, k)
    gt_expanded = expand_neighborhood_torch(gt_shoreline, k)
    
    intersection_1 = (pred_shoreline * gt_expanded).sum(dim=[1, 2])
    card_pred = pred_shoreline.sum(dim=[1, 2])
    card_gt_expanded = gt_expanded.sum(dim=[1,2]) # Use expanded card
    
    intersection_2 = (gt_shoreline * pred_expanded).sum(dim=[1, 2])
    card_gt = gt_shoreline.sum(dim=[1, 2])
    card_pred_expanded = pred_expanded.sum(dim=[1,2]) # Use expanded card
    
    eps = 1e-6
    scr_1 = (2.0 * intersection_1) / (card_pred + card_gt_expanded + eps)
    scr_2 = (2.0 * intersection_2) / (card_gt + card_pred_expanded + eps)
    scr_bidirectional = (scr_1 + scr_2) / 2.0
    return scr_bidirectional.mean()

class mSCRLoss(nn.Module):
    """Differentiable mSCR Loss"""
    def __init__(self, k_values=[4, 8, 24, 48]):
        super(mSCRLoss, self).__init__()
        self.k_values = k_values

    def forward(self, pred_logits, target):
        pred_probs = F.softmax(pred_logits, dim=1)
        pred_soft = pred_probs[:, 1, :, :] # Use Water class probability
        target_binary = (target > 0).float()
        scr_scores = []
        for k in self.k_values:
            pred_binary = (pred_soft > 0.5).float() # Binarize for shoreline extraction
            scr_k = compute_scr_differentiable(pred_binary, target_binary, k)
            scr_scores.append(scr_k)
        mscr = torch.stack(scr_scores).mean()
        loss = 1.0 - mscr
        return loss, mscr

# ============================================================================
# ### MODIFICATION: NEW SOTA LOSS FUNCTIONS (FOCAL + LOVASZ) ###
# ============================================================================

class FocalLoss(nn.Module):
    """
    Focal Loss, as described in https://arxiv.org/abs/1708.02002.
    It helps focus training on hard-to-classify examples (like boundaries).
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        # logits: (B, C, H, W), targets: (B, H, W)
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        
        # Get probabilities
        pt = torch.exp(-ce_loss)
        
        # Calculate Focal Loss
        focal_loss = (1 - pt)**self.gamma * ce_loss
        
        # Apply alpha weighting (optional)
        if self.alpha is not None:
            # Create alpha tensor for weighting
            alpha_t = torch.full_like(targets, 1 - self.alpha, dtype=torch.float32)
            alpha_t[targets == 1] = self.alpha # Apply alpha to class 1 (Water)
            focal_loss = alpha_t * focal_loss
            
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# ----------------------------------------------------------------------------
# Lovasz-Softmax Loss
# ----------------------------------------------------------------------------
# Code adapted from: https://github.com/bermanmaxim/LovaszSoftmax
# ----------------------------------------------------------------------------

def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted.float()).cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1: # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard

def lovasz_softmax_flat(probas, labels, classes='present'):
    """
    Multi-class Lovasz-Softmax loss
      probas: [P, C] Variable, class probabilities at each pixel (between 0 and 1)
      labels: [P] Variable, ground truth labels (between 0 and C-1)
      classes: 'all' for all, 'present' for classes present in labels
    """
    if probas.numel() == 0:
        # only void pixels, the gradients should be 0
        return probas * 0.
    C = probas.size(1)
    losses = []
    class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes
    for c in class_to_sum:
        fg = (labels == c).float() # foreground for class c
        if (classes == 'present' and fg.sum() == 0):
            continue
        if C == 1:
            if len(classes) > 1:
                raise ValueError('Sigmoid output possible only with 1 class')
            class_pred = probas[:, 0]
        else:
            class_pred = probas[:, c]
        errors = (fg - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, lovasz_grad(fg_sorted)))
    return torch.stack(losses).mean()


class LovaszSoftmax(nn.Module):
    """
    Multi-class Lovasz-Softmax loss.
    Directly optimizes the Jaccard index (IoU).
    """
    def __init__(self, classes='present', per_image=False, ignore_index=255):
        super(LovaszSoftmax, self).__init__()
        self.classes = classes
        self.per_image = per_image
        self.ignore_index = ignore_index

    def forward(self, logits, labels):
        # logits: (B, C, H, W), labels: (B, H, W)
        probas = F.softmax(logits, dim=1)
        
        if self.per_image:
            loss = torch.stack([
                lovasz_softmax_flat(
                    probas[i].permute(1, 2, 0).contiguous().view(-1, probas.size(1)),
                    labels[i].contiguous().view(-1)
                )
                for i in range(labels.size(0))
                if (labels[i] != self.ignore_index).any() # Only compute for images with valid pixels
            ]).mean()
        else:
            loss = lovasz_softmax_flat(
                probas.permute(0, 2, 3, 1).contiguous().view(-1, probas.size(1)),
                labels.contiguous().view(-1)
            )
        return loss

# ----------------------------------------------------------------------------
# End Lovasz-Softmax
# ----------------------------------------------------------------------------


class BoundaryAwareLoss(nn.Module):
    def __init__(self):
        super(BoundaryAwareLoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
    def forward(self, boundary_pred, target_mask):
        target_boundary = extract_shoreline_torch(target_mask)
        target_boundary = target_boundary.unsqueeze(1).float()
        loss = self.bce_loss(boundary_pred, target_boundary)
        return loss


class EnhancedCombinedLoss(nn.Module):
    """
    ### MODIFICATION: Enhanced Combined Loss with Focal + Lovasz ###
    """
    def __init__(self, weight_focal=1.0, weight_lovasz=1.0, weight_mscr=0.5, weight_boundary=0.3):
        super(EnhancedCombinedLoss, self).__init__()
        self.focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
        self.lovasz_loss = LovaszSoftmax()
        self.mscr_loss = mSCRLoss(k_values=cfg.K_NEIGHBORS)
        self.boundary_loss = BoundaryAwareLoss()

        self.weight_focal = weight_focal
        self.weight_lovasz = weight_lovasz
        self.weight_mscr = weight_mscr
        self.weight_boundary = weight_boundary

    def forward(self, pred, boundary_pred, target):
        focal = self.focal_loss(pred, target)
        lovasz = self.lovasz_loss(pred, target)
        mscr_loss, mscr_value = self.mscr_loss(pred, target)
        boundary = self.boundary_loss(boundary_pred, target)

        total_loss = (self.weight_focal * focal +
                      self.weight_lovasz * lovasz +
                      self.weight_mscr * mscr_loss +
                      self.weight_boundary * boundary)

        # Return itemized losses for logging
        return total_loss, (focal.item(), lovasz.item(), mscr_loss.item(), mscr_value.item(), boundary.item())

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
# POST-PROCESSING (Unchanged from v1)
# ============================================================================

def post_process_mask(pred_batch, min_area_threshold):
    """
    Cleans a batch of binary prediction masks by removing small connected components.
    """
    pred_np = pred_batch.cpu().numpy().astype(np.uint8)
    cleaned_masks_list = []
    
    for i in range(pred_np.shape[0]):
        mask = pred_np[i]
        
        # Find connected components in the *water* mask (class 1)
        num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )
        cleaned_mask = np.zeros_like(mask)
        
        # Iterate from label 1 (0 is the background 'land')
        for label in range(1, num_labels):
            area = stats[label, cv2.CC_STAT_AREA]
            if area > min_area_threshold:
                cleaned_mask[labels_im == label] = 1
        cleaned_masks_list.append(cleaned_mask)
        
    cleaned_batch_np = np.stack(cleaned_masks_list, axis=0)
    return torch.from_numpy(cleaned_batch_np).to(pred_batch.device)

# ============================================================================
# TRAINING & VALIDATION FUNCTIONS
# ============================================================================

def train_one_epoch(model, dataloader, criterion, optimizer, scaler, device, epoch):
    model.train()
    
    # ### MODIFICATION: Updated metric accumulators for new losses ###
    running_loss = 0.0
    running_focal_loss = 0.0
    running_lovasz_loss = 0.0
    running_mscr_loss = 0.0
    running_mscr_diff = 0.0
    running_boundary_loss = 0.0
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

        if cfg.USE_MIXED_PRECISION:
            with autocast():
                seg_outputs, boundary_outputs = model(images)
                loss, (focal_loss, lovasz_loss, mscr_loss, mscr_diff, boundary_loss) = criterion(
                    seg_outputs, boundary_outputs, masks
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            seg_outputs, boundary_outputs = model(images)
            loss, (focal_loss, lovasz_loss, mscr_loss, mscr_diff, boundary_loss) = criterion(
                seg_outputs, boundary_outputs, masks
            )
            loss.backward()
            optimizer.step()

        # Compute metrics
        pred = torch.argmax(seg_outputs, dim=1)
        acc = compute_accuracy(pred, masks)
        ious = compute_iou(pred, masks, num_classes=cfg.NUM_CLASSES)
        mscr_scores = compute_scr_by_k_for_batch(pred, masks, k_values=cfg.K_NEIGHBORS)

        # Update running metrics
        running_loss += loss.item()
        running_focal_loss += focal_loss
        running_lovasz_loss += lovasz_loss
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

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'mSCR_e': f'{mscr_scores["mSCR"]:.4f}',
            'focal': f'{focal_loss:.4f}'
        })

    num_batches = len(dataloader)
    metrics = {
        'loss': running_loss / num_batches,
        'focal_loss': running_focal_loss / num_batches,
        'lovasz_loss': running_lovasz_loss / num_batches,
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
    model.eval()
    
    # ### MODIFICATION: Updated metric accumulators for new losses ###
    running_loss = 0.0
    running_focal_loss = 0.0
    running_lovasz_loss = 0.0
    running_mscr_loss = 0.0
    running_mscr_diff = 0.0
    running_boundary_loss = 0.0
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
            loss, (focal_loss, lovasz_loss, mscr_loss, mscr_diff, boundary_loss) = criterion(
                seg_outputs, boundary_outputs, masks
            )

            # --- Apply Post-Processing ---
            pred_raw = torch.argmax(seg_outputs, dim=1)
            if cfg.USE_POST_PROCESSING:
                pred = post_process_mask(pred_raw, cfg.MIN_AREA_THRESHOLD)
            else:
                pred = pred_raw
            # --- End Post-Processing ---

            acc = compute_accuracy(pred, masks)
            ious = compute_iou(pred, masks, num_classes=cfg.NUM_CLASSES)
            mscr_scores = compute_scr_by_k_for_batch(pred, masks, k_values=cfg.K_NEIGHBORS)

            # Update running metrics
            running_loss += loss.item()
            running_focal_loss += focal_loss
            running_lovasz_loss += lovasz_loss
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

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'mSCR_e': f'{mscr_scores["mSCR"]:.4f}',
                'focal': f'{focal_loss:.4f}'
            })

    num_batches = len(dataloader)
    metrics = {
        'loss': running_loss / num_batches,
        'focal_loss': running_focal_loss / num_batches,
        'lovasz_loss': running_lovasz_loss / num_batches,
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
        # ### MODIFICATION: Updated plot lists for new losses ###
        self.train_losses = []
        self.val_losses = []
        self.train_focal_losses = []
        self.val_focal_losses = []
        self.train_lovasz_losses = []
        self.val_lovasz_losses = []
        self.train_mscr_losses = []
        self.val_mscr_losses = []
        self.train_boundary_losses = []
        self.val_boundary_losses = []
        self.train_accs = []
        self.val_accs = []
        self.train_ious = []
        self.val_ious = []
        self.train_mscr_diff = []
        self.val_mscr_diff = []
        self.train_mscr_eval = []
        self.val_mscr_eval = []
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
        self.train_losses.append(train_metrics['loss'])
        self.val_losses.append(val_metrics['loss'])
        self.train_focal_losses.append(train_metrics['focal_loss'])
        self.val_focal_losses.append(val_metrics['focal_loss'])
        self.train_lovasz_losses.append(train_metrics['lovasz_loss'])
        self.val_lovasz_losses.append(val_metrics['lovasz_loss'])
        self.train_mscr_losses.append(train_metrics['mscr_loss'])
        self.val_mscr_losses.append(val_metrics['mscr_loss'])
        self.train_boundary_losses.append(train_metrics['boundary_loss'])
        self.val_boundary_losses.append(val_metrics['boundary_loss'])
        self.train_accs.append(train_metrics['acc'])
        self.val_accs.append(val_metrics['acc'])
        self.train_ious.append(train_metrics['iou'])
        self.val_ious.append(val_metrics['iou'])
        self.train_mscr_diff.append(train_metrics['mscr_diff'])
        self.val_mscr_diff.append(val_metrics['mscr_diff'])
        self.train_mscr_eval.append(train_metrics['mscr_eval'])
        self.val_mscr_eval.append(val_metrics['mscr_eval'])
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
        plt.close('all')
        fig = plt.figure(figsize=(24, 24))
        gs = fig.add_gridspec(5, 2, hspace=0.35, wspace=0.25)
        epochs = range(1, len(self.train_losses) + 1)

        # Plot 1: Total Loss
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(epochs, self.train_losses, 'b-o', label='Train Loss', markersize=5)
        ax1.plot(epochs, self.val_losses, 'r-s', label='Val Loss', markersize=5)
        ax1.set_title('Total Loss (Focal + Lovasz + mSCR + Boundary)', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Loss Components (Focal + Lovasz)
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(epochs, self.val_focal_losses, 'g-^', label='Val Focal', markersize=4)
        ax2.plot(epochs, self.val_lovasz_losses, 'c-v', label='Val Lovasz', markersize=4)
        ax2.set_title('Validation Loss Components (Focal/Lovasz)', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot 3: Accuracy
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot(epochs, self.train_accs, 'b-o', label='Train Acc', markersize=5)
        ax3.plot(epochs, self.val_accs, 'r-s', label='Val Acc', markersize=5)
        ax3.set_title('Pixel-wise Accuracy', fontsize=14, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim([0.5, 1])

        # Plot 4: IoU
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.plot(epochs, self.train_ious, 'b-o', label='Train IoU', markersize=5)
        ax4.plot(epochs, self.val_ious, 'r-s', label='Val IoU', markersize=5)
        ax4.set_title('Intersection over Union', fontsize=14, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim([0.5, 1])

        # Plot 5: mSCR Differentiable vs Evaluation
        ax5 = fig.add_subplot(gs[2, 0])
        ax5.plot(epochs, self.val_mscr_diff, 'b-o', label='Val mSCR (Differentiable)', markersize=6)
        ax5.plot(epochs, self.val_mscr_eval, 'r-s', label='Val mSCR (Evaluation)', markersize=6)
        ax5.set_title('mSCR: Differentiable vs Evaluation', fontsize=14, fontweight='bold')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        ax5.set_ylim([0, 1])

        # Plot 6: SCR by Neighborhood Size (k)
        ax6 = fig.add_subplot(gs[2, 1])
        ax6.plot(epochs, self.val_mscr_eval, 'k-', label='Val mSCR (Mean)', alpha=0.7)
        ax6.plot(epochs, self.val_scr_k4s, 'g--o', label='Val SCR k=4', markersize=4, alpha=0.8)
        ax6.plot(epochs, self.val_scr_k8s, 'c--s', label='Val SCR k=8', markersize=4, alpha=0.8)
        ax6.plot(epochs, self.val_scr_k24s, 'm--^', label='Val SCR k=24', markersize=4, alpha=0.8)
        ax6.plot(epochs, self.val_scr_k48s, 'y--v', label='Val SCR k=48', markersize=4, alpha=0.8)
        ax6.set_title('SCR by Neighborhood Size (k)', fontsize=14, fontweight='bold')
        ax6.legend(fontsize=10)
        ax6.grid(True, alpha=0.3)
        ax6.set_ylim([0, 1])

        # Plot 7: mSCR Evaluation Metric (Primary)
        ax7 = fig.add_subplot(gs[3, 0])
        ax7.plot(epochs, self.train_mscr_eval, 'b-o', label='Train mSCR', markersize=6, alpha=0.7)
        ax7.plot(epochs, self.val_mscr_eval, 'r-s', label='Val mSCR', markersize=6)
        ax7.set_title('mSCR Evaluation Metric (Primary Metric)', fontsize=14, fontweight='bold')
        ax7.legend()
        ax7.grid(True, alpha=0.3)
        ax7.set_ylim([0, 1])

        # Plot 8: SCR k=4 vs k=48
        ax8 = fig.add_subplot(gs[3, 1])
        ax8.plot(epochs, self.train_scr_k4s, 'b:', label='Train k=4', alpha=0.6)
        ax8.plot(epochs, self.val_scr_k4s, 'g-o', label='Val k=4', markersize=5)
        ax8.plot(epochs, self.train_scr_k48s, 'c:', label='Train k=48', alpha=0.6)
        ax8.plot(epochs, self.val_scr_k48s, 'y-s', label='Val k=48', markersize=5)
        ax8.set_title('SCR k=4 vs k=48 (Strict vs Lenient)', fontsize=14, fontweight='bold')
        ax8.legend(fontsize=10)
        ax8.grid(True, alpha=0.3)
        ax8.set_ylim([0, 1])
        
        # Plot 9: Auxiliary Boundary Loss
        ax9 = fig.add_subplot(gs[4, 0])
        ax9.plot(epochs, self.train_boundary_losses, 'b-o', label='Train Boundary Loss', markersize=5)
        ax9.plot(epochs, self.val_boundary_losses, 'r-s', label='Val Boundary Loss', markersize=5)
        ax9.set_title('Auxiliary Boundary Head Loss', fontsize=14, fontweight='bold')
        ax9.legend()
        ax9.grid(True, alpha=0.3)

        # Plot 10: Other Loss Components (mSCR)
        ax10 = fig.add_subplot(gs[4, 1])
        ax10.plot(epochs, self.val_mscr_losses, 'm-d', label='Val mSCR Loss', markersize=4)
        ax10.set_title('Validation Loss Components (mSCR)', fontsize=14, fontweight='bold')
        ax10.legend()
        ax10.grid(True, alpha=0.3)
        
        # Save and display
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        clear_output(wait=True)
        display(IPImage(buf.read()))
        plt.close(fig)

    def save_final_plot(self, save_path):
        self._create_and_display_plot()
        # This is a bit of a hack to save the plot generated by _create...
        # A more robust way would be for _create... to return fig
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close('all')
        logger.info(f"Final training curves saved to: {save_path}")

# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================

def main():
    logger.info("=" * 80)
    logger.info("Enhanced HRNet-W48 (v2) - Focal + Lovasz + AdamW")
    logger.info("=" * 80)
    logger.info(f"Device: {cfg.DEVICE}")
    logger.info(f"Batch Size: {cfg.BATCH_SIZE}")
    logger.info(f"Number of Epochs: {cfg.NUM_EPOCHS}")
    logger.info(f"Learning Rate: {cfg.LEARNING_RATE}")
    logger.info(f"Optimizer: AdamW (Weight Decay: {cfg.WEIGHT_DECAY})")
    logger.info(f"Scheduler: CosineAnnealingLR (T_max: {cfg.NUM_EPOCHS})")
    logger.info(f"Mixed Precision: {cfg.USE_MIXED_PRECISION}")
    logger.info(f"Input Channels: {cfg.INPUT_CHANNELS} (RGB + Sobel + Canny)")
    logger.info(f"Loss Weights - Focal: {cfg.LOSS_WEIGHT_FOCAL}, Lovasz: {cfg.LOSS_WEIGHT_LOVASZ}, "
                f"mSCR: {cfg.LOSS_WEIGHT_MSCR}, Boundary: {cfg.LOSS_WEIGHT_BOUNDARY}")
    if cfg.USE_POST_PROCESSING:
        logger.info(f"POST-PROCESSING: ENABLED (Min Area: {cfg.MIN_AREA_THRESHOLD} pixels)")
    else:
        logger.info("POST-PROCESSING: DISABLED")
    logger.info("=" * 80)

    # Create datasets
    logger.info("\nLoading datasets...")
    train_dataset = CAIDHybridDataset(
        split_file=cfg.TRAIN_TXT, images_dir=cfg.IMAGES_DIR,
        masks_dir=cfg.MASKS_DIR, augment=True
    )
    val_dataset = CAIDHybridDataset(
        split_file=cfg.VAL_TXT, images_dir=cfg.IMAGES_DIR,
        masks_dir=cfg.MASKS_DIR, augment=False
    )
    train_loader = DataLoader(
        train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True,
        num_workers=cfg.NUM_WORKERS, pin_memory=cfg.PIN_MEMORY
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False,
        num_workers=cfg.NUM_WORKERS, pin_memory=cfg.PIN_MEMORY
    )
    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Val batches: {len(val_loader)}")

    # Create model
    logger.info("\nInitializing Enhanced HRNet-W48 with attention mechanisms...")
    model = HRNetW48(
        num_classes=cfg.NUM_CLASSES,
        input_channels=cfg.INPUT_CHANNELS
    ).to(cfg.DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    # ### MODIFICATION: New Loss, Optimizer, Scheduler ###
    criterion = EnhancedCombinedLoss(
        weight_focal=cfg.LOSS_WEIGHT_FOCAL,
        weight_lovasz=cfg.LOSS_WEIGHT_LOVASZ,
        weight_mscr=cfg.LOSS_WEIGHT_MSCR,
        weight_boundary=cfg.LOSS_WEIGHT_BOUNDARY
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.LEARNING_RATE,
        weight_decay=cfg.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.NUM_EPOCHS, eta_min=1e-6
    )
    scaler = GradScaler() if cfg.USE_MIXED_PRECISION else None
    
    plotter = MetricsPlotter()
    history = []
    best_val_mscr = 0.0

    logger.info("\n" + "=" * 80)
    logger.info("Starting Training (v2)...")
    logger.info("=" * 80)

    for epoch in range(cfg.NUM_EPOCHS):
        start_time = time.time()

        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, cfg.DEVICE, epoch
        )
        val_metrics = validate(
            model, val_loader, criterion, cfg.DEVICE, epoch
        )
        epoch_time = time.time() - start_time
        
        # ### MODIFICATION: Scheduler step (no metric needed) ###
        scheduler.step()

        # Log epoch summary
        logger.info("\n" + "=" * 80)
        logger.info(f"EPOCH {epoch + 1}/{cfg.NUM_EPOCHS} Summary (Time: {epoch_time:.2f}s) "
                    f"LR: {optimizer.param_groups[0]['lr']:.2e}")
        logger.info("=" * 80)
        logger.info(f"LOSSES:")
        logger.info(f"  Total     - Train: {train_metrics['loss']:.4f} | Val: {val_metrics['loss']:.4f}")
        logger.info(f"  Focal     - Train: {train_metrics['focal_loss']:.4f} | Val: {val_metrics['focal_loss']:.4f}")
        logger.info(f"  Lovasz    - Train: {train_metrics['lovasz_loss']:.4f} | Val: {val_metrics['lovasz_loss']:.4f}")
        logger.info(f"  mSCR Loss - Train: {train_metrics['mscr_loss']:.4f} | Val: {val_metrics['mscr_loss']:.4f}")
        logger.info(f"  Boundary  - Train: {train_metrics['boundary_loss']:.4f} | Val: {val_metrics['boundary_loss']:.4f}")
        logger.info(f"\nMETRICS:")
        logger.info(f"  Accuracy  - Train: {train_metrics['acc']:.4f} | Val: {val_metrics['acc']:.4f}")
        logger.info(f"  IoU       - Train: {train_metrics['iou']:.4f} | Val: {val_metrics['iou']:.4f}")
        logger.info(f"\nmSCR SCORES (Val mSCR is Post-Processed if enabled):")
        logger.info(f"  Evaluation - Train: {train_metrics['mscr_eval']:.4f} | Val: {val_metrics['mscr_eval']:.4f}")
        logger.info(f"\nSCR by K-values (Validation):")
        logger.info(f"  k=4:  {val_metrics['scr_k4']:.4f}")
        logger.info(f"  k=8:  {val_metrics['scr_k8']:.4f}")
        logger.info(f"  k=24: {val_metrics['scr_k24']:.4f}")
        logger.info(f"  k=48: {val_metrics['scr_k48']:.4f}")

        # Save to history
        history_entry = {
            'epoch': epoch + 1,
            'train_loss': train_metrics['loss'], 'val_loss': val_metrics['loss'],
            'train_focal_loss': train_metrics['focal_loss'], 'val_focal_loss': val_metrics['focal_loss'],
            'train_lovasz_loss': train_metrics['lovasz_loss'], 'val_lovasz_loss': val_metrics['lovasz_loss'],
            'train_mscr_loss': train_metrics['mscr_loss'], 'val_mscr_loss': val_metrics['mscr_loss'],
            'train_boundary_loss': train_metrics['boundary_loss'], 'val_boundary_loss': val_metrics['boundary_loss'],
            'train_acc': train_metrics['acc'], 'val_acc': val_metrics['acc'],
            'train_iou': train_metrics['iou'], 'val_iou': val_metrics['iou'],
            'train_mscr_diff': train_metrics['mscr_diff'], 'val_mscr_diff': val_metrics['mscr_diff'],
            'train_mscr_eval': train_metrics['mscr_eval'], 'val_mscr_eval': val_metrics['mscr_eval'],
            'train_scr_k4': train_metrics['scr_k4'], 'val_scr_k4': val_metrics['scr_k4'],
            'train_scr_k8': train_metrics['scr_k8'], 'val_scr_k8': val_metrics['scr_k8'],
            'train_scr_k24': train_metrics['scr_k24'], 'val_scr_k24': val_metrics['scr_k24'],
            'train_scr_k48': train_metrics['scr_k48'], 'val_scr_k48': val_metrics['scr_k48']
        }
        history.append(history_entry)

        plotter.update(train_metrics, val_metrics)

        if val_metrics['mscr_eval'] > best_val_mscr:
            best_val_mscr = val_metrics['mscr_eval']
            logger.info(f"New best validation mSCR ({best_val_mscr:.4f}). Saving model...")
            torch.save(model.state_dict(), cfg.BEST_MODEL_PATH)

    logger.info("\n" + "=" * 80)
    logger.info("Training Finished!")
    logger.info("=" * 80)
    logger.info(f"Best Validation mSCR (Post-Processed): {best_val_mscr:.4f}")
    logger.info(f"Best model saved to: {cfg.BEST_MODEL_PATH}")

    try:
        plotter.save_final_plot(os.path.join(cfg.OUTPUT_DIR, 'training_curves.png'))
    except Exception as e:
        logger.error(f"Could not save final plot: {e}")
        
    history_df = pd.DataFrame(history)
    history_df.to_csv(os.path.join(cfg.OUTPUT_DIR, 'training_history.csv'), index=False)
    logger.info(f"Training history saved to: {os.path.join(cfg.OUTPUT_DIR, 'training_history.csv')}")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logger.exception(f"An error occurred during main execution: {e}")
        raise e