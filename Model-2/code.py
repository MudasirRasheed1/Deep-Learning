"""
EncNet (Context Encoding Network) Training Script for CAID Dataset
Semantic Segmentation: Land vs Water Classification
With CORRECTED mSCR (mean Shoreline Conformity Rate) Evaluation Metric
and Differentiable mSCR Loss Function
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
from IPython.display import display, clear_output, Image as IPImage
from scipy.ndimage import binary_dilation
import io

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import torchvision.transforms as T
import torchvision.models as models

plt.ion()

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
    BEST_MODEL_PATH = os.path.join(OUTPUT_DIR, 'encnet_best.pth')
    
    # Training hyperparameters
    NUM_CLASSES = 2  # Land (0) and Water (1)
    BATCH_SIZE = 16
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    
    # Model
    BACKBONE = 'resnet101'
    PRETRAINED = True
    
    # Training options
    USE_MIXED_PRECISION = True
    NUM_WORKERS = 2
    PIN_MEMORY = True
    
    # Image settings
    IMAGE_SIZE = (500, 500)
    
    # Augmentation probabilities
    FLIP_PROB = 0.5
    ROTATE_PROB = 0.3
    
    # mSCR evaluation settings - CORRECTED: Now represents neighborhood sizes
    K_NEIGHBORS = [4, 8, 24, 48]  # Different neighborhood sizes (N values)
    
    # Loss weights
    LOSS_WEIGHT_CE = 1.0      # Cross-entropy loss weight
    LOSS_WEIGHT_DICE = 1.0    # Dice loss weight
    LOSS_WEIGHT_MSCR = 0.5    # mSCR loss weight (differentiable)
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

cfg = Config()

# ============================================================================
# DATASET CLASS
# ============================================================================

class CAIDDataset(Dataset):
    """
    Custom Dataset for CAID (Coastal Aerial Imagery Dataset)
    Loads images and segmentation masks for binary classification (Land vs Water)
    """
    
    def __init__(self, split_file, images_dir, masks_dir, transform=None, augment=False):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.augment = augment
        
        # Read image IDs from split file
        with open(split_file, 'r') as f:
            self.image_ids = [line.strip() for line in f.readlines()]
        
        print(f"Loaded {len(self.image_ids)} samples from {split_file}")
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        # Get image ID
        img_id = self.image_ids[idx]
        
        # Load image and mask (both are PNG format)
        img_path = os.path.join(self.images_dir, f"{img_id}.png")
        mask_path = os.path.join(self.masks_dir, f"{img_id}.png")
        
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')  # Grayscale
        
        # Apply augmentations (synchronized for image and mask)
        if self.augment:
            image, mask = self.apply_augmentation(image, mask)
        
        # Convert to tensor
        image = T.ToTensor()(image)
        mask = torch.from_numpy(np.array(mask)).long()
        
        # Normalize image (ImageNet stats)
        image = T.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])(image)
        
        return image, mask
    
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
# ENCNET MODEL COMPONENTS
# ============================================================================

class ContextEncoding(nn.Module):
    """
    Context Encoding Module
    Learns to encode semantic context from feature maps
    """
    
    def __init__(self, in_channels, num_codes):
        super(ContextEncoding, self).__init__()
        self.in_channels = in_channels
        self.num_codes = num_codes
        
        # Learnable codewords and scaling factors
        self.codewords = nn.Parameter(torch.randn(num_codes, in_channels))
        self.scale = nn.Parameter(torch.randn(num_codes))
        
    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        
        # Reshape features: (B, C, H*W) -> (B, H*W, C)
        features = x.view(B, C, -1).permute(0, 2, 1)  # (B, N, C)
        
        # Compute assignment weights (soft assignment to codewords)
        codewords = self.codewords.unsqueeze(0)  # (1, K, C)
        
        # Compute L2 distance: (B, N, K)
        dist = torch.cdist(features, codewords)
        
        # Soft assignment using scaled softmax
        assignment = F.softmax(-dist * self.scale.view(1, 1, -1), dim=2)  # (B, N, K)
        
        # Aggregate residuals: weighted sum of (features - codewords)
        residuals = features.unsqueeze(2) - codewords.unsqueeze(1)  # (B, N, K, C)
        encoded = torch.sum(assignment.unsqueeze(3) * residuals, dim=1)  # (B, K, C)
        
        # Global pooling of encoded features
        encoded = encoded.mean(dim=1)  # (B, C)
        
        return encoded


class EncModule(nn.Module):
    """
    Encoding Module: combines context encoding with SE (Squeeze-and-Excitation)
    """
    
    def __init__(self, in_channels, num_codes):
        super(EncModule, self).__init__()
        
        self.encoding = ContextEncoding(in_channels, num_codes)
        
        # Fully connected layers for channel attention (SE block)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // 4, in_channels),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Encode context
        encoded = self.encoding(x)  # (B, C)
        
        # Generate channel attention weights
        attention = self.fc(encoded).unsqueeze(2).unsqueeze(3)  # (B, C, 1, 1)
        
        # Apply attention to input features
        out = x * attention
        
        return out, encoded


class EncNet(nn.Module):
    """
    Context Encoding Network (EncNet) for Semantic Segmentation
    Uses ResNet backbone + Context Encoding Module + Segmentation Head
    """
    
    def __init__(self, num_classes=2, backbone='resnet50', pretrained=True):
        super(EncNet, self).__init__()
        
        # Load pretrained ResNet backbone
        if backbone == 'resnet50':
            resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
        elif backbone == 'resnet101':
            resnet = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1 if pretrained else None)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Extract backbone layers
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        
        self.layer1 = resnet.layer1  # 256 channels
        self.layer2 = resnet.layer2  # 512 channels
        self.layer3 = resnet.layer3  # 1024 channels
        self.layer4 = resnet.layer4  # 2048 channels
        
        # Context Encoding Module
        self.enc_module = EncModule(in_channels=2048, num_codes=32)
        
        # Segmentation head
        self.seg_head = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, num_classes, kernel_size=1)
        )
        
    def forward(self, x):
        input_size = x.size()[2:]
        
        # Backbone forward pass
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)  # (B, 2048, H/32, W/32)
        
        # Apply context encoding
        x, encoded = self.enc_module(x)
        
        # Segmentation head
        out = self.seg_head(x)
        
        # Upsample to original size
        out = F.interpolate(out, size=input_size, mode='bilinear', align_corners=False)
        
        return out, encoded

# ============================================================================
# DIFFERENTIABLE mSCR LOSS FUNCTION
# ============================================================================

def extract_shoreline_torch(mask):
    """
    Extract shoreline (boundary) pixels from a binary mask using differentiable operations.
    
    Args:
        mask: torch.Tensor of shape (B, H, W) with values 0 or 1 (after argmax)
    
    Returns:
        Shoreline mask as torch.Tensor (B, H, W) with boundary pixels
    """
    # Convert to float for convolution operations
    mask_float = mask.float().unsqueeze(1)  # (B, 1, H, W)
    
    # Create Laplacian kernel for edge detection (detects boundaries)
    # This kernel will highlight regions where there's a change in values
    laplacian_kernel = torch.tensor([
        [1, 1, 1],
        [1, -8, 1],
        [1, 1, 1]
    ], dtype=torch.float32, device=mask.device).view(1, 1, 3, 3)
    
    # Apply convolution to detect edges (boundaries between land and water)
    edges = F.conv2d(mask_float, laplacian_kernel, padding=1)
    
    # Shoreline pixels are where edges exist (non-zero values)
    # Use absolute value and threshold to get binary shoreline mask
    shoreline = (edges.abs() > 0.1).float().squeeze(1)  # (B, H, W)
    
    return shoreline


def expand_neighborhood_torch(shoreline, k):
    """
    Expand shoreline pixels by k-neighborhood using max pooling (differentiable).
    
    Args:
        shoreline: torch.Tensor (B, H, W) - binary shoreline mask
        k: int - neighborhood size (4, 8, 24, 48)
    
    Returns:
        Expanded shoreline mask (B, H, W)
    """
    # Map k values to kernel sizes for max pooling
    # k=4 -> 3x3, k=8 -> 3x3, k=24 -> 5x5, k=48 -> 7x7
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
    
    # Add channel dimension for max pooling
    shoreline_expanded = shoreline.unsqueeze(1)  # (B, 1, H, W)
    
    # Use max pooling to expand the neighborhood (acts like dilation)
    # Padding ensures output size matches input size
    padding = kernel_size // 2
    expanded = F.max_pool2d(shoreline_expanded, kernel_size=kernel_size, 
                           stride=1, padding=padding)
    
    return expanded.squeeze(1)  # (B, H, W)


def compute_scr_differentiable(pred_mask, gt_mask, k):
    """
    Compute differentiable SCR for a single k value.
    
    SCR_k^N = (2 × |S1 ∩ S(S2, N)|) / (|S1| + |S2|)
    
    Args:
        pred_mask: Predicted mask (B, H, W) - after argmax, values 0 or 1
        gt_mask: Ground truth mask (B, H, W) - values 0 or 1
        k: Neighborhood size
    
    Returns:
        SCR score averaged across batch (scalar tensor)
    """
    batch_size = pred_mask.shape[0]
    
    # Extract shorelines for predicted and ground truth masks
    pred_shoreline = extract_shoreline_torch(pred_mask)  # (B, H, W)
    gt_shoreline = extract_shoreline_torch(gt_mask)      # (B, H, W)
    
    # Expand neighborhoods
    pred_expanded = expand_neighborhood_torch(pred_shoreline, k)  # (B, H, W)
    gt_expanded = expand_neighborhood_torch(gt_shoreline, k)      # (B, H, W)
    
    # Compute intersection and cardinalities for each image in batch
    # Direction 1: pred shoreline -> expanded gt
    intersection_1 = (pred_shoreline * gt_expanded).sum(dim=[1, 2])  # (B,)
    card_pred = pred_shoreline.sum(dim=[1, 2])  # (B,)
    card_gt = gt_shoreline.sum(dim=[1, 2])      # (B,)
    
    # Direction 2: gt shoreline -> expanded pred
    intersection_2 = (gt_shoreline * pred_expanded).sum(dim=[1, 2])  # (B,)
    
    # Compute SCR for both directions (with epsilon to avoid division by zero)
    eps = 1e-6
    scr_1 = (2.0 * intersection_1) / (card_pred + card_gt + eps)  # (B,)
    scr_2 = (2.0 * intersection_2) / (card_gt + card_pred + eps)  # (B,)
    
    # Average both directions (bidirectional SCR)
    scr_bidirectional = (scr_1 + scr_2) / 2.0  # (B,)
    
    # Return mean across batch
    return scr_bidirectional.mean()


class mSCRLoss(nn.Module):
    """
    Differentiable mSCR Loss Function.
    
    The loss is computed as: Loss = 1 - mSCR
    where mSCR is averaged across both images and k-neighborhood values.
    
    According to the paper:
    mSCR = (1/n) * Σ(SCR_I) where SCR_I is averaged across k values for image I
    """
    
    def __init__(self, k_values=[4, 8, 24, 48]):
        super(mSCRLoss, self).__init__()
        self.k_values = k_values
        
    def forward(self, pred_logits, target):
        """
        Compute mSCR loss.
        
        Args:
            pred_logits: Predicted logits (B, C, H, W)
            target: Ground truth masks (B, H, W) with class indices
        
        Returns:
            mSCR loss (scalar)
        """
        # Convert logits to class predictions (soft approximation for differentiability)
        # Use softmax + weighted sum instead of argmax for gradient flow
        pred_probs = F.softmax(pred_logits, dim=1)  # (B, C, H, W)
        
        # For binary segmentation, use probability of water class (class 1)
        # as soft prediction mask
        pred_soft = pred_probs[:, 1, :, :]  # (B, H, W) - probability of water
        
        # Convert target to binary (0 or 1)
        target_binary = (target > 0).float()  # (B, H, W)
        
        # Compute SCR for each k value
        scr_scores = []
        for k in self.k_values:
            # For differentiability, we use soft predictions
            # In practice, we threshold soft predictions to get binary masks
            pred_binary = (pred_soft > 0.5).float()
            
            # Compute SCR for this k value (returns mean across batch)
            scr_k = compute_scr_differentiable(pred_binary, target_binary, k)
            scr_scores.append(scr_k)
        
        # CORRECTED: Average across k values to get mean SCR per image
        # Then it's already averaged across images in compute_scr_differentiable
        mscr = torch.stack(scr_scores).mean()
        
        # Return loss as (1 - mSCR) so that minimizing loss maximizes mSCR
        loss = 1.0 - mscr
        
        return loss

# ============================================================================
# TRADITIONAL LOSS FUNCTIONS
# ============================================================================

class DiceLoss(nn.Module):
    """Dice Loss for binary segmentation"""
    
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        # pred: (B, C, H, W) - logits
        # target: (B, H, W) - class indices
        
        pred = torch.softmax(pred, dim=1)
        
        # Convert target to one-hot
        target_one_hot = F.one_hot(target, num_classes=pred.shape[1]).permute(0, 3, 1, 2).float()
        
        # Flatten
        pred = pred.contiguous().view(-1)
        target_one_hot = target_one_hot.contiguous().view(-1)
        
        intersection = (pred * target_one_hot).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target_one_hot.sum() + self.smooth)
        
        return 1 - dice


class CombinedLoss(nn.Module):
    """
    Combined Loss: Cross-Entropy + Dice + mSCR Loss
    
    Total Loss = w_ce * CE + w_dice * Dice + w_mscr * (1 - mSCR)
    """
    
    def __init__(self, weight_ce=1.0, weight_dice=1.0, weight_mscr=0.5):
        super(CombinedLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss()
        self.mscr_loss = mSCRLoss(k_values=cfg.K_NEIGHBORS)
        
        self.weight_ce = weight_ce
        self.weight_dice = weight_dice
        self.weight_mscr = weight_mscr
        
    def forward(self, pred, target):
        """
        Compute combined loss.
        
        Returns:
            total_loss, (ce_loss, dice_loss, mscr_loss) for logging
        """
        # Compute individual losses
        ce = self.ce_loss(pred, target)
        dice = self.dice_loss(pred, target)
        mscr = self.mscr_loss(pred, target)
        
        # Weighted combination
        total_loss = (self.weight_ce * ce + 
                     self.weight_dice * dice + 
                     self.weight_mscr * mscr)
        
        # Return total loss and individual components for logging
        return total_loss, (ce.item(), dice.item(), mscr.item())

# ============================================================================
# CORRECTED mSCR METRIC EVALUATION (Non-differentiable, for validation)
# ============================================================================

def extract_shoreline_pixels(mask):
    """
    Extract shoreline (boundary) pixels from a binary mask (numpy version).
    """
    binary_mask = (mask > 0).astype(np.uint8)
    
    # Dilate to find boundaries
    dilated = binary_dilation(binary_mask, structure=np.ones((3, 3)))
    eroded = binary_dilation(1 - binary_mask, structure=np.ones((3, 3)))
    
    # Shoreline is boundary region
    shoreline = dilated & eroded
    
    return shoreline


def expand_neighborhood(shoreline_pixels, k):
    """
    Expand shoreline pixels by k-neighborhood (numpy version).
    """
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
    """
    Compute SCR for a single direction (numpy version).
    """
    S1 = np.sum(shoreline_1)
    S2_expanded = np.sum(shoreline_2_expanded)
    
    if S1 == 0 and S2_expanded == 0:
        return 1.0
    if S1 == 0 or S2_expanded == 0:
        return 0.0
    
    intersection = np.sum(shoreline_1 & shoreline_2_expanded)
    scr = (2.0 * intersection) / (S1 + S2_expanded)
    
    return scr


def compute_scr_for_image(pred_mask, gt_mask, k_values=[4, 8, 24, 48]):
    """
    CORRECTED: Compute SCR for a SINGLE image across all k values.
    Returns the average SCR across k values for this image.
    
    This is SCR_I in the formula: SCR_I = (1/|K|) * Σ(SCR_k)
    """
    # Extract shoreline pixels
    pred_shoreline = extract_shoreline_pixels(pred_mask)
    gt_shoreline = extract_shoreline_pixels(gt_mask)
    
    scr_scores = []
    
    for k in k_values:
        # Expand neighborhoods
        pred_expanded = expand_neighborhood(pred_shoreline, k)
        gt_expanded = expand_neighborhood(gt_shoreline, k)
        
        # Compute bidirectional SCR
        scr_pred_to_gt = compute_scr_single(pred_shoreline, gt_expanded)
        scr_gt_to_pred = compute_scr_single(gt_shoreline, pred_expanded)
        
        # Average both directions
        scr_k = (scr_pred_to_gt + scr_gt_to_pred) / 2.0
        scr_scores.append(scr_k)
    
    # Return average SCR across all k values for this image
    return np.mean(scr_scores)


def compute_mscr_batch(pred_batch, target_batch, k_values=[4, 8, 24, 48]):
    """
    CORRECTED: Compute mSCR for a batch following the paper's formula:
    
    mSCR = (1/n) * Σ(SCR_I) where I = 1 to n images
    
    Each SCR_I is the average across k values for image I.
    """
    # Convert to numpy if tensor
    if torch.is_tensor(pred_batch):
        pred_batch = pred_batch.cpu().numpy()
    if torch.is_tensor(target_batch):
        target_batch = target_batch.cpu().numpy()
    
    batch_size = pred_batch.shape[0]
    
    # Compute SCR for each image (averaged across k values)
    scr_per_image = []
    for i in range(batch_size):
        scr_i = compute_scr_for_image(pred_batch[i], target_batch[i], k_values)
        scr_per_image.append(scr_i)
    
    # CORRECTED: mSCR is the average across images
    mscr = np.mean(scr_per_image)
    
    # Also return individual k-value statistics for monitoring
    # (averaged across all images in the batch)
    scr_k_stats = {f'SCR_k{k}': 0.0 for k in k_values}
    
    for i in range(batch_size):
        pred_shoreline = extract_shoreline_pixels(pred_batch[i])
        gt_shoreline = extract_shoreline_pixels(target_batch[i])
        
        for k in k_values:
            pred_expanded = expand_neighborhood(pred_shoreline, k)
            gt_expanded = expand_neighborhood(gt_shoreline, k)
            
            scr_pred_to_gt = compute_scr_single(pred_shoreline, gt_expanded)
            scr_gt_to_pred = compute_scr_single(gt_shoreline, pred_expanded)
            scr_k = (scr_pred_to_gt + scr_gt_to_pred) / 2.0
            
            scr_k_stats[f'SCR_k{k}'] += scr_k
    
    # Average k-statistics across batch
    for k in k_values:
        scr_k_stats[f'SCR_k{k}'] /= batch_size
    
    results = {'mSCR': mscr}
    results.update(scr_k_stats)
    
    return results

# ============================================================================
# TRADITIONAL METRICS
# ============================================================================

def compute_iou(pred, target, num_classes=2):
    """Compute Intersection over Union (IoU) for each class"""
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
    """Train for one epoch with detailed loss tracking"""
    
    model.train()
    
    # Metric accumulators
    running_loss = 0.0
    running_ce_loss = 0.0
    running_dice_loss = 0.0
    running_mscr_loss = 0.0
    running_acc = 0.0
    running_iou_land = 0.0
    running_iou_water = 0.0
    running_mscr = 0.0
    running_scr_k4 = 0.0
    running_scr_k8 = 0.0
    running_scr_k24 = 0.0
    running_scr_k48 = 0.0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} [TRAIN]")
    
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision training
        if cfg.USE_MIXED_PRECISION:
            with autocast():
                outputs, encoded = model(images)
                # Get total loss and individual loss components
                loss, (ce_loss, dice_loss, mscr_loss) = criterion(outputs, masks)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs, encoded = model(images)
            loss, (ce_loss, dice_loss, mscr_loss) = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
        
        # Compute metrics for monitoring
        pred = torch.argmax(outputs, dim=1)
        
        acc = compute_accuracy(pred, masks)
        ious = compute_iou(pred, masks, num_classes=cfg.NUM_CLASSES)
        
        # Compute mSCR metric (evaluation version, non-differentiable)
        mscr_scores = compute_mscr_batch(pred, masks, k_values=cfg.K_NEIGHBORS)
        
        # Update running metrics
        running_loss += loss.item()
        running_ce_loss += ce_loss
        running_dice_loss += dice_loss
        running_mscr_loss += mscr_loss
        running_acc += acc
        running_iou_land += ious[0] if not np.isnan(ious[0]) else 0
        running_iou_water += ious[1] if not np.isnan(ious[1]) else 0
        running_mscr += mscr_scores['mSCR']
        running_scr_k4 += mscr_scores['SCR_k4']
        running_scr_k8 += mscr_scores['SCR_k8']
        running_scr_k24 += mscr_scores['SCR_k24']
        running_scr_k48 += mscr_scores['SCR_k48']
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'ce': f'{ce_loss:.4f}',
            'dice': f'{dice_loss:.4f}',
            'mscr_l': f'{mscr_loss:.4f}',
            'mSCR': f'{mscr_scores["mSCR"]:.4f}'
        })
    
    # Average metrics across all batches
    num_batches = len(dataloader)
    avg_loss = running_loss / num_batches
    avg_ce_loss = running_ce_loss / num_batches
    avg_dice_loss = running_dice_loss / num_batches
    avg_mscr_loss = running_mscr_loss / num_batches
    avg_acc = running_acc / num_batches
    avg_iou_land = running_iou_land / num_batches
    avg_iou_water = running_iou_water / num_batches
    avg_iou = (avg_iou_land + avg_iou_water) / 2
    avg_mscr = running_mscr / num_batches
    avg_scr_k4 = running_scr_k4 / num_batches
    avg_scr_k8 = running_scr_k8 / num_batches
    avg_scr_k24 = running_scr_k24 / num_batches
    avg_scr_k48 = running_scr_k48 / num_batches
    
    return (avg_loss, avg_ce_loss, avg_dice_loss, avg_mscr_loss,
            avg_acc, avg_iou, avg_iou_land, avg_iou_water,
            avg_mscr, avg_scr_k4, avg_scr_k8, avg_scr_k24, avg_scr_k48)


def validate(model, dataloader, criterion, device, epoch):
    """Validate the model with detailed loss tracking"""
    
    model.eval()
    
    # Metric accumulators
    running_loss = 0.0
    running_ce_loss = 0.0
    running_dice_loss = 0.0
    running_mscr_loss = 0.0
    running_acc = 0.0
    running_iou_land = 0.0
    running_iou_water = 0.0
    running_mscr = 0.0
    running_scr_k4 = 0.0
    running_scr_k8 = 0.0
    running_scr_k24 = 0.0
    running_scr_k48 = 0.0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} [VAL]  ")
    
    with torch.no_grad():
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)
            
            outputs, encoded = model(images)
            loss, (ce_loss, dice_loss, mscr_loss) = criterion(outputs, masks)
            
            # Compute metrics
            pred = torch.argmax(outputs, dim=1)
            
            acc = compute_accuracy(pred, masks)
            ious = compute_iou(pred, masks, num_classes=cfg.NUM_CLASSES)
            
            # Compute mSCR metric (evaluation version)
            mscr_scores = compute_mscr_batch(pred, masks, k_values=cfg.K_NEIGHBORS)
            
            # Update running metrics
            running_loss += loss.item()
            running_ce_loss += ce_loss
            running_dice_loss += dice_loss
            running_mscr_loss += mscr_loss
            running_acc += acc
            running_iou_land += ious[0] if not np.isnan(ious[0]) else 0
            running_iou_water += ious[1] if not np.isnan(ious[1]) else 0
            running_mscr += mscr_scores['mSCR']
            running_scr_k4 += mscr_scores['SCR_k4']
            running_scr_k8 += mscr_scores['SCR_k8']
            running_scr_k24 += mscr_scores['SCR_k24']
            running_scr_k48 += mscr_scores['SCR_k48']
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'ce': f'{ce_loss:.4f}',
                'dice': f'{dice_loss:.4f}',
                'mscr_l': f'{mscr_loss:.4f}',
                'mSCR': f'{mscr_scores["mSCR"]:.4f}'
            })
    
    # Average metrics
    num_batches = len(dataloader)
    avg_loss = running_loss / num_batches
    avg_ce_loss = running_ce_loss / num_batches
    avg_dice_loss = running_dice_loss / num_batches
    avg_mscr_loss = running_mscr_loss / num_batches
    avg_acc = running_acc / num_batches
    avg_iou_land = running_iou_land / num_batches
    avg_iou_water = running_iou_water / num_batches
    avg_iou = (avg_iou_land + avg_iou_water) / 2
    avg_mscr = running_mscr / num_batches
    avg_scr_k4 = running_scr_k4 / num_batches
    avg_scr_k8 = running_scr_k8 / num_batches
    avg_scr_k24 = running_scr_k24 / num_batches
    avg_scr_k48 = running_scr_k48 / num_batches
    
    return (avg_loss, avg_ce_loss, avg_dice_loss, avg_mscr_loss,
            avg_acc, avg_iou, avg_iou_land, avg_iou_water,
            avg_mscr, avg_scr_k4, avg_scr_k8, avg_scr_k24, avg_scr_k48)

# ============================================================================
# ENHANCED VISUALIZATION WITH LOSS COMPONENTS
# ============================================================================

class MetricsPlotter:
    """
    Enhanced real-time metrics plotting for Kaggle notebooks.
    Now includes detailed loss component visualization.
    """
    
    def __init__(self):
        # Loss components
        self.train_losses = []
        self.val_losses = []
        self.train_ce_losses = []
        self.val_ce_losses = []
        self.train_dice_losses = []
        self.val_dice_losses = []
        self.train_mscr_losses = []
        self.val_mscr_losses = []
        
        # Performance metrics
        self.train_accs = []
        self.val_accs = []
        self.train_ious = []
        self.val_ious = []
        self.train_mscrs = []
        self.val_mscrs = []
        
        # Individual k-value SCR scores
        self.train_scr_k4s = []
        self.val_scr_k4s = []
        self.train_scr_k8s = []
        self.val_scr_k8s = []
        self.train_scr_k24s = []
        self.val_scr_k24s = []
        self.train_scr_k48s = []
        self.val_scr_k48s = []
    
    def update(self, train_metrics, val_metrics):
        """Update plots with new metrics - Enhanced with loss components"""
        
        # Unpack all metrics from the tuples (13 values each)
        (train_loss, train_ce, train_dice, train_mscr_l,
         train_acc, train_iou, _, _,
         train_mscr, train_scr_k4, train_scr_k8, train_scr_k24, train_scr_k48) = train_metrics
        
        (val_loss, val_ce, val_dice, val_mscr_l,
         val_acc, val_iou, _, _,
         val_mscr, val_scr_k4, val_scr_k8, val_scr_k24, val_scr_k48) = val_metrics

        # Append loss components
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.train_ce_losses.append(train_ce)
        self.val_ce_losses.append(val_ce)
        self.train_dice_losses.append(train_dice)
        self.val_dice_losses.append(val_dice)
        self.train_mscr_losses.append(train_mscr_l)
        self.val_mscr_losses.append(val_mscr_l)
        
        # Append performance metrics
        self.train_accs.append(train_acc)
        self.val_accs.append(val_acc)
        self.train_ious.append(train_iou)
        self.val_ious.append(val_iou)
        self.train_mscrs.append(train_mscr)
        self.val_mscrs.append(val_mscr)

        # Append individual k-values
        self.train_scr_k4s.append(train_scr_k4)
        self.val_scr_k4s.append(val_scr_k4)
        self.train_scr_k8s.append(train_scr_k8)
        self.val_scr_k8s.append(val_scr_k8)
        self.train_scr_k24s.append(train_scr_k24)
        self.val_scr_k24s.append(val_scr_k24)
        self.train_scr_k48s.append(train_scr_k48)
        self.val_scr_k48s.append(val_scr_k48)
        
        # Create and display plot
        self._create_and_display_plot()
    
    def _create_and_display_plot(self):
        """Create enhanced plot with loss components and display it properly in Kaggle"""
        
        # Close any existing figures
        plt.close('all')
        
        # Create new figure with 6 subplots (3x2 grid)
        fig = plt.figure(figsize=(20, 18))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.25)
        
        epochs = range(1, len(self.train_losses) + 1)
        
        # ===== Subplot 1: Total Loss =====
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(epochs, self.train_losses, 'b-o', label='Train Total Loss', linewidth=2, markersize=5)
        ax1.plot(epochs, self.val_losses, 'r-s', label='Val Total Loss', linewidth=2, markersize=5)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Total Loss (CE + Dice + mSCR)', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # ===== Subplot 2: Loss Components Breakdown =====
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(epochs, self.val_ce_losses, 'g-^', label='Val CE Loss', linewidth=2, markersize=4)
        ax2.plot(epochs, self.val_dice_losses, 'c-v', label='Val Dice Loss', linewidth=2, markersize=4)
        ax2.plot(epochs, self.val_mscr_losses, 'm-d', label='Val mSCR Loss', linewidth=2, markersize=4)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Loss Value', fontsize=12)
        ax2.set_title('Validation Loss Components', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        
        # ===== Subplot 3: Accuracy =====
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot(epochs, self.train_accs, 'b-o', label='Train Acc', linewidth=2, markersize=5)
        ax3.plot(epochs, self.val_accs, 'r-s', label='Val Acc', linewidth=2, markersize=5)
        ax3.set_xlabel('Epoch', fontsize=12)
        ax3.set_ylabel('Accuracy', fontsize=12)
        ax3.set_title('Pixel-wise Accuracy', fontsize=14, fontweight='bold')
        ax3.legend(fontsize=11)
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim([0, 1])
        
        # ===== Subplot 4: IoU =====
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.plot(epochs, self.train_ious, 'b-o', label='Train IoU', linewidth=2, markersize=5)
        ax4.plot(epochs, self.val_ious, 'r-s', label='Val IoU', linewidth=2, markersize=5)
        ax4.set_xlabel('Epoch', fontsize=12)
        ax4.set_ylabel('Mean IoU', fontsize=12)
        ax4.set_title('Intersection over Union', fontsize=14, fontweight='bold')
        ax4.legend(fontsize=11)
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim([0, 1])
        
        # ===== Subplot 5: mSCR Metric (Evaluation) =====
        ax5 = fig.add_subplot(gs[2, 0])
        ax5.plot(epochs, self.train_mscrs, 'b-o', label='Train mSCR', linewidth=2.5, markersize=6, alpha=0.7)
        ax5.plot(epochs, self.val_mscrs, 'r-s', label='Val mSCR', linewidth=2.5, markersize=6)
        ax5.set_xlabel('Epoch', fontsize=12)
        ax5.set_ylabel('mSCR Score', fontsize=12)
        ax5.set_title('mSCR Metric (Avg across images)', fontsize=14, fontweight='bold')
        ax5.legend(fontsize=11)
        ax5.grid(True, alpha=0.3)
        ax5.set_ylim([0, 1])
        
        # ===== Subplot 6: Individual k-value SCR Scores =====
        ax6 = fig.add_subplot(gs[2, 1])
        ax6.plot(epochs, self.val_mscrs, 'r-s', label='Val mSCR (Mean)', linewidth=2.5, markersize=6)
        ax6.plot(epochs, self.val_scr_k4s, 'g:', label='Val SCR k=4', linewidth=1.5, alpha=0.8)
        ax6.plot(epochs, self.val_scr_k8s, 'c:', label='Val SCR k=8', linewidth=1.5, alpha=0.8)
        ax6.plot(epochs, self.val_scr_k24s, 'm:', label='Val SCR k=24', linewidth=1.5, alpha=0.8)
        ax6.plot(epochs, self.val_scr_k48s, 'y:', label='Val SCR k=48', linewidth=1.5, alpha=0.8)
        ax6.set_xlabel('Epoch', fontsize=12)
        ax6.set_ylabel('SCR Score', fontsize=12)
        ax6.set_title('SCR by Neighborhood Size (k)', fontsize=14, fontweight='bold')
        ax6.legend(fontsize=10)
        ax6.grid(True, alpha=0.3)
        ax6.set_ylim([0, 1])
        
        # Save to buffer and display
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        
        # Clear output and display new plot
        clear_output(wait=True)
        display(IPImage(buf.read()))
        
        plt.close(fig)
        
    def save_final_plot(self, save_path):
        """Save the final training curves with all metrics"""
        
        # Close any existing figures
        plt.close('all')
        
        fig = plt.figure(figsize=(20, 18))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.25)
        
        epochs = range(1, len(self.train_losses) + 1)
        
        # ===== Subplot 1: Total Loss =====
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(epochs, self.train_losses, 'b-o', label='Train Total Loss', linewidth=2, markersize=5)
        ax1.plot(epochs, self.val_losses, 'r-s', label='Val Total Loss', linewidth=2, markersize=5)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Total Loss (CE + Dice + mSCR)', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # ===== Subplot 2: Loss Components =====
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(epochs, self.train_ce_losses, 'b:', label='Train CE', linewidth=1.5, alpha=0.6)
        ax2.plot(epochs, self.val_ce_losses, 'g-^', label='Val CE Loss', linewidth=2, markersize=4)
        ax2.plot(epochs, self.train_dice_losses, 'c:', label='Train Dice', linewidth=1.5, alpha=0.6)
        ax2.plot(epochs, self.val_dice_losses, 'c-v', label='Val Dice Loss', linewidth=2, markersize=4)
        ax2.plot(epochs, self.train_mscr_losses, 'm:', label='Train mSCR Loss', linewidth=1.5, alpha=0.6)
        ax2.plot(epochs, self.val_mscr_losses, 'm-d', label='Val mSCR Loss', linewidth=2, markersize=4)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Loss Value', fontsize=12)
        ax2.set_title('Loss Components Breakdown', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)
        
        # ===== Subplot 3: Accuracy =====
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot(epochs, self.train_accs, 'b-o', label='Train Acc', linewidth=2, markersize=5)
        ax3.plot(epochs, self.val_accs, 'r-s', label='Val Acc', linewidth=2, markersize=5)
        ax3.set_xlabel('Epoch', fontsize=12)
        ax3.set_ylabel('Accuracy', fontsize=12)
        ax3.set_title('Pixel-wise Accuracy', fontsize=14, fontweight='bold')
        ax3.legend(fontsize=11)
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim([0, 1])
        
        # ===== Subplot 4: IoU =====
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.plot(epochs, self.train_ious, 'b-o', label='Train IoU', linewidth=2, markersize=5)
        ax4.plot(epochs, self.val_ious, 'r-s', label='Val IoU', linewidth=2, markersize=5)
        ax4.set_xlabel('Epoch', fontsize=12)
        ax4.set_ylabel('Mean IoU', fontsize=12)
        ax4.set_title('Intersection over Union', fontsize=14, fontweight='bold')
        ax4.legend(fontsize=11)
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim([0, 1])
        
        # ===== Subplot 5: mSCR Metric =====
        ax5 = fig.add_subplot(gs[2, 0])
        ax5.plot(epochs, self.train_mscrs, 'b-o', label='Train mSCR', linewidth=2.5, markersize=6, alpha=0.7)
        ax5.plot(epochs, self.val_mscrs, 'r-s', label='Val mSCR', linewidth=2.5, markersize=6)
        ax5.set_xlabel('Epoch', fontsize=12)
        ax5.set_ylabel('mSCR Score', fontsize=12)
        ax5.set_title('mSCR Metric (Avg across images)', fontsize=14, fontweight='bold')
        ax5.legend(fontsize=11)
        ax5.grid(True, alpha=0.3)
        ax5.set_ylim([0, 1])
        
        # ===== Subplot 6: Individual k-values =====
        ax6 = fig.add_subplot(gs[2, 1])
        ax6.plot(epochs, self.val_mscrs, 'r-s', label='Val mSCR (Mean)', linewidth=2.5, markersize=6)
        ax6.plot(epochs, self.val_scr_k4s, 'g:', label='Val SCR k=4', linewidth=1.5, alpha=0.8)
        ax6.plot(epochs, self.val_scr_k8s, 'c:', label='Val SCR k=8', linewidth=1.5, alpha=0.8)
        ax6.plot(epochs, self.val_scr_k24s, 'm:', label='Val SCR k=24', linewidth=1.5, alpha=0.8)
        ax6.plot(epochs, self.val_scr_k48s, 'y:', label='Val SCR k=48', linewidth=1.5, alpha=0.8)
        ax6.set_xlabel('Epoch', fontsize=12)
        ax6.set_ylabel('SCR Score', fontsize=12)
        ax6.set_title('SCR by Neighborhood Size (k)', fontsize=14, fontweight='bold')
        ax6.legend(fontsize=10)
        ax6.grid(True, alpha=0.3)
        ax6.set_ylim([0, 1])
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"✓ Final training curves saved to: {save_path}")

# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================
def main():
    """Main training function with enhanced metrics tracking"""
    
    print("=" * 80)
    print("EncNet Training on CAID Dataset")
    print("With CORRECTED mSCR Evaluation & Differentiable mSCR Loss")
    print("=" * 80)
    print(f"Device: {cfg.DEVICE}")
    print(f"Batch Size: {cfg.BATCH_SIZE}")
    print(f"Number of Epochs: {cfg.NUM_EPOCHS}")
    print(f"Learning Rate: {cfg.LEARNING_RATE}")
    print(f"Mixed Precision: {cfg.USE_MIXED_PRECISION}")
    print(f"K-Neighbors for mSCR: {cfg.K_NEIGHBORS}")
    print(f"Loss Weights - CE: {cfg.LOSS_WEIGHT_CE}, Dice: {cfg.LOSS_WEIGHT_DICE}, mSCR: {cfg.LOSS_WEIGHT_MSCR}")
    print("=" * 80)
    
    # Create datasets
    print("\nLoading datasets...")
    train_dataset = CAIDDataset(
        split_file=cfg.TRAIN_TXT,
        images_dir=cfg.IMAGES_DIR,
        masks_dir=cfg.MASKS_DIR,
        augment=True
    )
    
    val_dataset = CAIDDataset(
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
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Create model
    print("\nInitializing EncNet model...")
    model = EncNet(
        num_classes=cfg.NUM_CLASSES,
        backbone=cfg.BACKBONE,
        pretrained=cfg.PRETRAINED
    ).to(cfg.DEVICE)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss function and optimizer
    criterion = CombinedLoss(
        weight_ce=cfg.LOSS_WEIGHT_CE,
        weight_dice=cfg.LOSS_WEIGHT_DICE,
        weight_mscr=cfg.LOSS_WEIGHT_MSCR
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.LEARNING_RATE,
        weight_decay=cfg.WEIGHT_DECAY
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Mixed precision scaler
    scaler = GradScaler() if cfg.USE_MIXED_PRECISION else None
    
    # Metrics plotter
    plotter = MetricsPlotter()
    
    # Training history
    history = []
    
    # Best model tracking
    best_val_iou = 0.0
    best_val_mscr = 0.0
    
    print("\n" + "=" * 80)
    print("Starting Training...")
    print("=" * 80)
    
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
        
        # Extract metrics (13 values each)
        (train_loss, train_ce, train_dice, train_mscr_l,
         train_acc, train_iou, train_iou_land, train_iou_water,
         train_mscr, train_scr_k4, train_scr_k8, train_scr_k24, train_scr_k48) = train_metrics
        
        (val_loss, val_ce, val_dice, val_mscr_l,
         val_acc, val_iou, val_iou_land, val_iou_water,
         val_mscr, val_scr_k4, val_scr_k8, val_scr_k24, val_scr_k48) = val_metrics
        
        # Update learning rate scheduler based on validation loss
        scheduler.step(val_loss)
        
        # Print epoch summary
        print("\n" + "=" * 80)
        print(f"EPOCH {epoch + 1}/{cfg.NUM_EPOCHS} Summary (Time: {epoch_time:.2f}s)")
        print("=" * 80)
        print(f"LOSSES:")
        print(f"  Total    - Train: {train_loss:.4f} | Val: {val_loss:.4f}")
        print(f"  CE       - Train: {train_ce:.4f} | Val: {val_ce:.4f}")
        print(f"  Dice     - Train: {train_dice:.4f} | Val: {val_dice:.4f}")
        print(f"  mSCR Loss- Train: {train_mscr_l:.4f} | Val: {val_mscr_l:.4f}")
        print(f"\nMETRICS:")
        print(f"  Accuracy - Train: {train_acc:.4f} | Val: {val_acc:.4f}")
        print(f"  IoU      - Train: {train_iou:.4f} | Val: {val_iou:.4f}")
        print(f"    Land   - Train: {train_iou_land:.4f} | Val: {val_iou_land:.4f}")
        print(f"    Water  - Train: {train_iou_water:.4f} | Val: {val_iou_water:.4f}")
        print(f"  mSCR     - Train: {train_mscr:.4f} | Val: {val_mscr:.4f}")
        print(f"\nSCR by K-values (Validation):")
        print(f"  k=4:  {val_scr_k4:.4f}")
        print(f"  k=8:  {val_scr_k8:.4f}")
        print(f"  k=24: {val_scr_k24:.4f}")
        print(f"  k=48: {val_scr_k48:.4f}")
        
        # Save to history
        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_ce': train_ce,
            'train_dice': train_dice,
            'train_mscr_loss': train_mscr_l,
            'train_acc': train_acc,
            'train_iou': train_iou,
            'train_iou_land': train_iou_land,
            'train_iou_water': train_iou_water,
            'train_mscr': train_mscr,
            'val_loss': val_loss,
            'val_ce': val_ce,
            'val_dice': val_dice,
            'val_mscr_loss': val_mscr_l,
            'val_acc': val_acc,
            'val_iou': val_iou,
            'val_iou_land': val_iou_land,
            'val_iou_water': val_iou_water,
            'val_mscr': val_mscr,
            'val_scr_k4': val_scr_k4,
            'val_scr_k8': val_scr_k8,
            'val_scr_k24': val_scr_k24,
            'val_scr_k48': val_scr_k48,
            'epoch_time': epoch_time
        })
        
        # Update live plots
        plotter.update(train_metrics, val_metrics)
        
        # Save best model based on validation mSCR
        if val_mscr > best_val_mscr:
            best_val_mscr = val_mscr
            best_val_iou = val_iou
            
            # Save model checkpoint
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_mscr': val_mscr,
                'val_iou': val_iou,
                'val_acc': val_acc,
                'config': {
                    'num_classes': cfg.NUM_CLASSES,
                    'backbone': cfg.BACKBONE,
                    'image_size': cfg.IMAGE_SIZE,
                    'k_neighbors': cfg.K_NEIGHBORS
                }
            }
            
            torch.save(checkpoint, cfg.BEST_MODEL_PATH)
            print(f"\n✓ New best model saved! Val mSCR: {val_mscr:.4f}, Val IoU: {val_iou:.4f}")
        
        print("=" * 80 + "\n")
    
    # Training complete
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    print(f"Best Validation mSCR: {best_val_mscr:.4f}")
    print(f"Best Validation IoU: {best_val_iou:.4f}")
    print(f"Best model saved at: {cfg.BEST_MODEL_PATH}")
    
    # Save final training curves
    final_plot_path = os.path.join(cfg.OUTPUT_DIR, 'training_curves.png')
    plotter.save_final_plot(final_plot_path)
    
    # Save training history to CSV
    history_df = pd.DataFrame(history)
    history_csv_path = os.path.join(cfg.OUTPUT_DIR, 'training_history.csv')
    history_df.to_csv(history_csv_path, index=False)
    print(f"✓ Training history saved to: {history_csv_path}")
    
    # Print final summary statistics
    print("\n" + "=" * 80)
    print("TRAINING STATISTICS")
    print("=" * 80)
    print(f"Total training time: {sum([h['epoch_time'] for h in history]) / 60:.2f} minutes")
    print(f"Average epoch time: {np.mean([h['epoch_time'] for h in history]):.2f} seconds")
    print(f"\nFinal Training Metrics:")
    print(f"  Loss: {train_loss:.4f}")
    print(f"  Accuracy: {train_acc:.4f}")
    print(f"  IoU: {train_iou:.4f}")
    print(f"  mSCR: {train_mscr:.4f}")
    print(f"\nFinal Validation Metrics:")
    print(f"  Loss: {val_loss:.4f}")
    print(f"  Accuracy: {val_acc:.4f}")
    print(f"  IoU: {val_iou:.4f}")
    print(f"  mSCR: {val_mscr:.4f}")
    print("=" * 80)
    
    return model, history


# ============================================================================
# SCRIPT ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    # Run training
    model, history = main()
    
    print("\n✓ All done! Check the output directory for results.")