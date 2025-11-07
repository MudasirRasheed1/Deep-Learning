"""
Seg-GAN (Adversarial Segmentation Network) for CAID Dataset
Semantic Segmentation: Land vs Water Classification
Using U-Net++ (EfficientNet-B5) Generator and a PatchGAN Discriminator.
Optimizes for both segmentation accuracy and boundary realism.
"""

# Install required libraries - FIX numpy version compatibility
# !pip install -q segmentation-models-pytorch albumentations
# !pip install -q "numpy<2.0" --force-reinstall

import os
import numpy as np
import pandas as pd
import cv2
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
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

plt.ion()

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    # --- Paths ---
    DATA_ROOT = '/kaggle/input/caid-dataset/CAID'
    IMAGES_DIR = os.path.join(DATA_ROOT, 'JPEGImages')
    MASKS_DIR = os.path.join(DATA_ROOT, 'SegmentationClass')
    TRAIN_TXT = os.path.join(DATA_ROOT, 'ImageSets/Segmentation/train.txt')
    VAL_TXT = os.path.join(DATA_ROOT, 'ImageSets/Segmentation/val.txt')
    TEST_TXT = os.path.join(DATA_ROOT, 'ImageSets/Segmentation/test.txt')
    
    # --- Output ---
    OUTPUT_DIR = '/kaggle/working'
    BEST_MODEL_PATH = os.path.join(OUTPUT_DIR, 'seg_gan_generator_best.pth')
    
    # --- Model & Training Hyperparameters ---
    NUM_CLASSES = 2  # Land (0) and Water (1)
    BATCH_SIZE = 8
    NUM_EPOCHS = 12  # Changed from 30 to 12
    IMAGE_SIZE = (512, 512)
    
    # --- Learning Rates for GANs ---
    LR_GENERATOR = 1e-4
    LR_DISCRIMINATOR = 1e-4
    B1 = 0.5 
    B2 = 0.999
    
    # --- Generator (Segmentation Model) ---
    GENERATOR_ENCODER = 'efficientnet-b5'
    GENERATOR_PRETRAINED = 'imagenet'
    
    # --- Training options ---
    USE_MIXED_PRECISION = True
    NUM_WORKERS = 2
    PIN_MEMORY = True
    
    # --- mSCR evaluation settings ---
    K_NEIGHBORS = [4, 8, 24, 48]
    
    # --- Loss Weights for the Generator ---
    LOSS_WEIGHT_CE = 1.0
    LOSS_WEIGHT_DICE = 1.0
    LOSS_WEIGHT_MSCR = 0.5
    LOSS_WEIGHT_ADVERSARIAL = 0.1

    # --- Device ---
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

cfg = Config()

# ============================================================================
# DATASET & AUGMENTATIONS
# ============================================================================

class CAIDDataset(Dataset):
    """CAID Dataset with resizing and augmentations"""
    
    def __init__(self, split_file, images_dir, masks_dir, augment=False):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.augment = augment
        
        with open(split_file, 'r') as f:
            self.image_ids = [line.strip() for line in f.readlines()]
            
        print(f"Loaded {len(self.image_ids)} samples from {split_file}")

        # Validation transforms
        self.val_transform = A.Compose([
            A.Resize(height=cfg.IMAGE_SIZE[0], width=cfg.IMAGE_SIZE[1], interpolation=cv2.INTER_NEAREST),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
        
        # Training transforms
        self.train_transform = A.Compose([
            A.Resize(height=cfg.IMAGE_SIZE[0], width=cfg.IMAGE_SIZE[1], interpolation=cv2.INTER_NEAREST),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=35, p=0.3),
            A.RandomBrightnessContrast(p=0.3),
            A.GaussNoise(p=0.2),
            A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        
        img_path = os.path.join(self.images_dir, f"{img_id}.png")
        mask_path = os.path.join(self.masks_dir, f"{img_id}.png")
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if self.augment:
            transformed = self.train_transform(image=image, mask=mask)
        else:
            transformed = self.val_transform(image=image, mask=mask)
            
        image = transformed['image']
        mask = transformed['mask'].long()
        
        return image, mask

# ============================================================================
# MODELS
# ============================================================================

def build_generator(device=cfg.DEVICE):
    """Builds the Generator model (U-Net++)"""
    model = smp.UnetPlusPlus(
        encoder_name=cfg.GENERATOR_ENCODER,
        encoder_weights=cfg.GENERATOR_PRETRAINED,
        in_channels=3,
        classes=cfg.NUM_CLASSES
    )
    return model.to(device)

class Discriminator(nn.Module):
    """PatchGAN Discriminator"""
    def __init__(self, in_channels=5):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [
                nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True)
            ]
            if bn:
                block.append(nn.BatchNorm2d(out_filters))
            return block

        self.model = nn.Sequential(
            *discriminator_block(in_channels, 64, bn=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1)
        )

    def forward(self, img, mask_logits):
        img_input = torch.cat((img, mask_logits), dim=1)
        return self.model(img_input)

def build_discriminator(device=cfg.DEVICE):
    model = Discriminator(in_channels=cfg.NUM_CLASSES + 3)
    return model.to(device)

# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

def extract_shoreline_torch(mask):
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
    if k == 4: kernel_size = 3
    elif k == 8: kernel_size = 3
    elif k == 24: kernel_size = 5
    elif k == 48: kernel_size = 7
    else: kernel_size = 3
    shoreline_expanded = shoreline.unsqueeze(1)
    padding = kernel_size // 2
    expanded = F.max_pool2d(shoreline_expanded, kernel_size=kernel_size,
                            stride=1, padding=padding)
    return expanded.squeeze(1)

def compute_scr_differentiable(pred_mask, gt_mask, k):
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
    """Differentiable mSCR Loss Function"""
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
        return loss

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

class CombinedGeneratorLoss(nn.Module):
    """Combined Loss for the Generator"""
    
    def __init__(self, weight_ce=1.0, weight_dice=1.0, weight_mscr=0.5):
        super(CombinedGeneratorLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss()
        self.mscr_loss = mSCRLoss(k_values=cfg.K_NEIGHBORS)
        
        self.weight_ce = weight_ce
        self.weight_dice = weight_dice
        self.weight_mscr = weight_mscr
        
    def forward(self, pred, target):
        ce = self.ce_loss(pred, target)
        dice = self.dice_loss(pred, target)
        mscr = self.mscr_loss(pred, target)
        
        total_loss = (self.weight_ce * ce + 
                      self.weight_dice * dice + 
                      self.weight_mscr * mscr)
        
        return total_loss, (ce.item(), dice.item(), mscr.item())

class AdversarialLoss(nn.Module):
    """Adversarial Loss (LSGAN)"""
    def __init__(self):
        super(AdversarialLoss, self).__init__()
        self.loss_fn = nn.MSELoss()

    def forward(self, output, is_real):
        if is_real:
            target = torch.ones_like(output, device=output.device)
        else:
            target = torch.zeros_like(output, device=output.device)
            
        return self.loss_fn(output, target)

# ============================================================================
# EVALUATION METRICS (Non-differentiable for validation)
# ============================================================================

def extract_shoreline_pixels(mask):
    binary_mask = (mask > 0).astype(np.uint8)
    dilated = binary_dilation(binary_mask, structure=np.ones((3, 3)))
    eroded = binary_dilation(1 - binary_mask, structure=np.ones((3, 3)))
    shoreline = dilated & eroded
    return shoreline

def expand_neighborhood(shoreline_pixels, k):
    if k == 4: structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
    elif k == 8: structure = np.ones((3, 3), dtype=np.uint8)
    elif k == 24: structure = np.ones((5, 5), dtype=np.uint8)
    elif k == 48: structure = np.ones((7, 7), dtype=np.uint8)
    else: raise ValueError(f"Unsupported k value: {k}")
    expanded = binary_dilation(shoreline_pixels.astype(np.uint8), structure=structure)
    return expanded

def compute_scr_single(shoreline_1, shoreline_2_expanded):
    S1 = np.sum(shoreline_1)
    S2_expanded = np.sum(shoreline_2_expanded)
    
    if S1 == 0 and S2_expanded == 0: return 1.0
    if S1 == 0 or S2_expanded == 0: return 0.0
    
    intersection = np.sum(shoreline_1 & shoreline_2_expanded)
    scr = (2.0 * intersection) / (S1 + S2_expanded)
    return scr

def compute_scr_for_image(pred_mask, gt_mask, k_values=[4, 8, 24, 48]):
    pred_shoreline = extract_shoreline_pixels(pred_mask)
    gt_shoreline = extract_shoreline_pixels(gt_mask)
    scr_scores = []
    
    for k in k_values:
        pred_expanded = expand_neighborhood(pred_shoreline, k)
        gt_expanded = expand_neighborhood(gt_shoreline, k)
        scr_pred_to_gt = compute_scr_single(pred_shoreline, gt_expanded)
        scr_gt_to_pred = compute_scr_single(gt_shoreline, pred_expanded)
        scr_k = (scr_pred_to_gt + scr_gt_to_pred) / 2.0
        scr_scores.append(scr_k)
        
    return np.mean(scr_scores)

def compute_mscr_batch(pred_batch, target_batch, k_values=[4, 8, 24, 48]):
    if torch.is_tensor(pred_batch): pred_batch = pred_batch.cpu().numpy()
    if torch.is_tensor(target_batch): target_batch = target_batch.cpu().numpy()
    
    batch_size = pred_batch.shape[0]
    scr_per_image = []
    for i in range(batch_size):
        scr_i = compute_scr_for_image(pred_batch[i], target_batch[i], k_values)
        scr_per_image.append(scr_i)
        
    mscr = np.mean(scr_per_image)
    
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
            
    for k in k_values:
        scr_k_stats[f'SCR_k{k}'] /= batch_size
        
    results = {'mSCR': mscr}
    results.update(scr_k_stats)
    return results

def compute_iou(pred, target, num_classes=2):
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
    correct = (pred == target).sum().item()
    total = target.numel()
    return correct / total

# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def train_one_epoch(model_G, model_D, dataloader,
                    opt_G, opt_D,
                    loss_seg, loss_adv,
                    scaler_G, scaler_D,
                    device, epoch):
    
    model_G.train()
    model_D.train()
    
    running_loss_G_total = 0.0
    running_loss_G_seg = 0.0
    running_loss_G_adv = 0.0
    running_loss_D_total = 0.0
    
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
    
    for images, gt_masks in pbar:
        images = images.to(device)
        gt_masks = gt_masks.to(device)
        
        # --- Train Discriminator ---
        model_D.requires_grad_(True)
        opt_D.zero_grad()
        
        with autocast(enabled=cfg.USE_MIXED_PRECISION):
            fake_logits = model_G(images).detach()
            fake_output = model_D(images, fake_logits)
            loss_D_fake = loss_adv(fake_output, is_real=False)

            gt_masks_one_hot = F.one_hot(gt_masks, cfg.NUM_CLASSES).permute(0, 3, 1, 2).float()
            real_output = model_D(images, gt_masks_one_hot)
            loss_D_real = loss_adv(real_output, is_real=True)
            
            loss_D = (loss_D_fake + loss_D_real) * 0.5

        scaler_D.scale(loss_D).backward()
        scaler_D.step(opt_D)
        scaler_D.update()

        # --- Train Generator ---
        model_D.requires_grad_(False)
        opt_G.zero_grad()
        
        with autocast(enabled=cfg.USE_MIXED_PRECISION):
            fake_logits = model_G(images)
            loss_G_seg_components, (ce_loss, dice_loss, mscr_loss) = loss_seg(fake_logits, gt_masks)
            
            fake_output = model_D(images, fake_logits)
            loss_G_adv = loss_adv(fake_output, is_real=True)
            
            loss_G = (loss_G_seg_components + 
                      cfg.LOSS_WEIGHT_ADVERSARIAL * loss_G_adv)

        scaler_G.scale(loss_G).backward()
        scaler_G.step(opt_G)
        scaler_G.update()
        
        # --- Accumulate Metrics ---
        running_loss_G_total += loss_G.item()
        running_loss_G_seg += loss_G_seg_components.item()
        running_loss_G_adv += loss_G_adv.item()
        running_loss_D_total += loss_D.item()
        
        running_ce_loss += ce_loss
        running_dice_loss += dice_loss
        running_mscr_loss += mscr_loss
        
        pred = torch.argmax(fake_logits, dim=1)
        acc = compute_accuracy(pred, gt_masks)
        ious = compute_iou(pred, gt_masks, num_classes=cfg.NUM_CLASSES)
        mscr_scores = compute_mscr_batch(pred, gt_masks, k_values=cfg.K_NEIGHBORS)
        
        running_acc += acc
        running_iou_land += ious[0] if not np.isnan(ious[0]) else 0
        running_iou_water += ious[1] if not np.isnan(ious[1]) else 0
        running_mscr += mscr_scores['mSCR']
        running_scr_k4 += mscr_scores['SCR_k4']
        running_scr_k8 += mscr_scores['SCR_k8']
        running_scr_k24 += mscr_scores['SCR_k24']
        running_scr_k48 += mscr_scores['SCR_k48']
        
        pbar.set_postfix({
            'L_G': f'{loss_G.item():.4f}',
            'L_D': f'{loss_D.item():.4f}',
            'L_adv': f'{loss_G_adv.item():.4f}',
            'mSCR': f'{mscr_scores["mSCR"]:.4f}'
        })

    num_batches = len(dataloader)
    avg_loss_G_total = running_loss_G_total / num_batches
    avg_loss_G_seg = running_loss_G_seg / num_batches
    avg_loss_G_adv = running_loss_G_adv / num_batches
    avg_loss_D_total = running_loss_D_total / num_batches
    
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

    return (avg_loss_G_seg, avg_ce_loss, avg_dice_loss, avg_mscr_loss,
            avg_acc, avg_iou, avg_iou_land, avg_iou_water,
            avg_mscr, avg_scr_k4, avg_scr_k8, avg_scr_k24, avg_scr_k48,
            avg_loss_G_total, avg_loss_G_adv, avg_loss_D_total)

# ============================================================================
# VALIDATION FUNCTION
# ============================================================================

def validate(model_G, dataloader, criterion_seg, device, epoch):
    """Validate the model (only Generator)"""
    
    model_G.eval()
    
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
            
            outputs = model_G(images)
            
            loss, (ce_loss, dice_loss, mscr_loss) = criterion_seg(outputs, masks)
            
            pred = torch.argmax(outputs, dim=1)
            
            acc = compute_accuracy(pred, masks)
            ious = compute_iou(pred, masks, num_classes=cfg.NUM_CLASSES)
            mscr_scores = compute_mscr_batch(pred, masks, k_values=cfg.K_NEIGHBORS)
            
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
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'mSCR': f'{mscr_scores["mSCR"]:.4f}'
            })

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
# VISUALIZATION
# ============================================================================

class MetricsPlotter:
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
        self.train_mscrs = []
        self.val_mscrs = []
        self.train_scr_k4s = []
        self.val_scr_k4s = []
        self.train_scr_k8s = []
        self.val_scr_k8s = []
        self.train_scr_k24s = []
        self.val_scr_k24s = []
        self.train_scr_k48s = []
        self.val_scr_k48s = []
    
    def update(self, train_metrics, val_metrics):
        (train_loss, train_ce, train_dice, train_mscr_l,
         train_acc, train_iou, _, _,
         train_mscr, train_scr_k4, train_scr_k8, train_scr_k24, train_scr_k48) = train_metrics
        
        (val_loss, val_ce, val_dice, val_mscr_l,
         val_acc, val_iou, _, _,
         val_mscr, val_scr_k4, val_scr_k8, val_scr_k24, val_scr_k48) = val_metrics

        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.train_ce_losses.append(train_ce)
        self.val_ce_losses.append(val_ce)
        self.train_dice_losses.append(train_dice)
        self.val_dice_losses.append(val_dice)
        self.train_mscr_losses.append(train_mscr_l)
        self.val_mscr_losses.append(val_mscr_l)
        self.train_accs.append(train_acc)
        self.val_accs.append(val_acc)
        self.train_ious.append(train_iou)
        self.val_ious.append(val_iou)
        self.train_mscrs.append(train_mscr)
        self.val_mscrs.append(val_mscr)
        self.train_scr_k4s.append(train_scr_k4)
        self.val_scr_k4s.append(val_scr_k4)
        self.train_scr_k8s.append(train_scr_k8)
        self.val_scr_k8s.append(val_scr_k8)
        self.train_scr_k24s.append(train_scr_k24)
        self.val_scr_k24s.append(val_scr_k24)
        self.train_scr_k48s.append(train_scr_k48)
        self.val_scr_k48s.append(val_scr_k48)
        
        self._create_and_display_plot()
    
    def _create_and_display_plot(self):
        plt.close('all')
        fig = plt.figure(figsize=(20, 18))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.25)
        epochs = range(1, len(self.train_losses) + 1)
        
        # Subplot 1: Total Loss
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(epochs, self.train_losses, 'b-o', label='Train Seg Loss', linewidth=2, markersize=5)
        ax1.plot(epochs, self.val_losses, 'r-s', label='Val Seg Loss', linewidth=2, markersize=5)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Segmentation Loss (CE + Dice + mSCR)', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # Subplot 2: Loss Components Breakdown
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(epochs, self.val_ce_losses, 'g-^', label='Val CE Loss', linewidth=2, markersize=4)
        ax2.plot(epochs, self.val_dice_losses, 'c-v', label='Val Dice Loss', linewidth=2, markersize=4)
        ax2.plot(epochs, self.val_mscr_losses, 'm-d', label='Val mSCR Loss', linewidth=2, markersize=4)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Loss Value', fontsize=12)
        ax2.set_title('Validation Loss Components', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        
        # Subplot 3: Accuracy
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot(epochs, self.train_accs, 'b-o', label='Train Acc', linewidth=2, markersize=5)
        ax3.plot(epochs, self.val_accs, 'r-s', label='Val Acc', linewidth=2, markersize=5)
        ax3.set_xlabel('Epoch', fontsize=12)
        ax3.set_ylabel('Accuracy', fontsize=12)
        ax3.set_title('Pixel-wise Accuracy', fontsize=14, fontweight='bold')
        ax3.legend(fontsize=11)
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim([0.5, 1])
        
        # Subplot 4: IoU
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.plot(epochs, self.train_ious, 'b-o', label='Train IoU', linewidth=2, markersize=5)
        ax4.plot(epochs, self.val_ious, 'r-s', label='Val IoU', linewidth=2, markersize=5)
        ax4.set_xlabel('Epoch', fontsize=12)
        ax4.set_ylabel('Mean IoU', fontsize=12)
        ax4.set_title('Intersection over Union', fontsize=14, fontweight='bold')
        ax4.legend(fontsize=11)
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim([0.5, 1])
        
        # Subplot 5: mSCR Metric
        ax5 = fig.add_subplot(gs[2, 0])
        ax5.plot(epochs, self.train_mscrs, 'b-o', label='Train mSCR', linewidth=2.5, markersize=6, alpha=0.7)
        ax5.plot(epochs, self.val_mscrs, 'r-s', label='Val mSCR', linewidth=2.5, markersize=6)
        ax5.set_xlabel('Epoch', fontsize=12)
        ax5.set_ylabel('mSCR Score', fontsize=12)
        ax5.set_title('mSCR Metric (Avg across images)', fontsize=14, fontweight='bold')
        ax5.legend(fontsize=11)
        ax5.grid(True, alpha=0.3)
        ax5.set_ylim([0, 1])
        
        # Subplot 6: Individual k-value SCR Scores
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
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        clear_output(wait=True)
        display(IPImage(buf.read()))
        plt.close(fig)
        
    def save_final_plot(self, save_path):
        plt.close('all')
        fig = plt.figure(figsize=(20, 18))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.25)
        epochs = range(1, len(self.train_losses) + 1)
        
        # Subplot 1: Total Loss
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(epochs, self.train_losses, 'b-o', label='Train Seg Loss', linewidth=2, markersize=5)
        ax1.plot(epochs, self.val_losses, 'r-s', label='Val Seg Loss', linewidth=2, markersize=5)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Segmentation Loss (CE + Dice + mSCR)', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # Subplot 2: Loss Components
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
        
        # Subplot 3: Accuracy
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot(epochs, self.train_accs, 'b-o', label='Train Acc', linewidth=2, markersize=5)
        ax3.plot(epochs, self.val_accs, 'r-s', label='Val Acc', linewidth=2, markersize=5)
        ax3.set_xlabel('Epoch', fontsize=12)
        ax3.set_ylabel('Accuracy', fontsize=12)
        ax3.set_title('Pixel-wise Accuracy', fontsize=14, fontweight='bold')
        ax3.legend(fontsize=11)
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim([0.5, 1])
        
        # Subplot 4: IoU
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.plot(epochs, self.train_ious, 'b-o', label='Train IoU', linewidth=2, markersize=5)
        ax4.plot(epochs, self.val_ious, 'r-s', label='Val IoU', linewidth=2, markersize=5)
        ax4.set_xlabel('Epoch', fontsize=12)
        ax4.set_ylabel('Mean IoU', fontsize=12)
        ax4.set_title('Intersection over Union', fontsize=14, fontweight='bold')
        ax4.legend(fontsize=11)
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim([0.5, 1])
        
        # Subplot 5: mSCR Metric
        ax5 = fig.add_subplot(gs[2, 0])
        ax5.plot(epochs, self.train_mscrs, 'b-o', label='Train mSCR', linewidth=2.5, markersize=6, alpha=0.7)
        ax5.plot(epochs, self.val_mscrs, 'r-s', label='Val mSCR', linewidth=2.5, markersize=6)
        ax5.set_xlabel('Epoch', fontsize=12)
        ax5.set_ylabel('mSCR Score', fontsize=12)
        ax5.set_title('mSCR Metric (Avg across images)', fontsize=14, fontweight='bold')
        ax5.legend(fontsize=11)
        ax5.grid(True, alpha=0.3)
        ax5.set_ylim([0, 1])
        
        # Subplot 6: Individual k-values
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
    print("=" * 80)
    print("Seg-GAN (Adversarial) Training on CAID Dataset")
    print("Generator: U-Net++ (EfficientNet-B5)")
    print("Discriminator: PatchGAN")
    print("=" * 80)
    print(f"Device: {cfg.DEVICE}")
    print(f"Batch Size: {cfg.BATCH_SIZE}")
    print(f"Image Size: {cfg.IMAGE_SIZE}")
    print(f"Number of Epochs: {cfg.NUM_EPOCHS}")
    print(f"K-Neighbors for mSCR: {cfg.K_NEIGHBORS}")
    print(f"Generator Loss Weights - CE: {cfg.LOSS_WEIGHT_CE}, Dice: {cfg.LOSS_WEIGHT_DICE}, mSCR: {cfg.LOSS_WEIGHT_MSCR}, Adv: {cfg.LOSS_WEIGHT_ADVERSARIAL}")
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
        train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True,
        num_workers=cfg.NUM_WORKERS, pin_memory=cfg.PIN_MEMORY
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False,
        num_workers=cfg.NUM_WORKERS, pin_memory=cfg.PIN_MEMORY
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Create models
    print("\nInitializing Generator (G) and Discriminator (D)...")
    model_G = build_generator(cfg.DEVICE)
    model_D = build_discriminator(cfg.DEVICE)
    
    # Count parameters
    g_params = sum(p.numel() for p in model_G.parameters() if p.requires_grad)
    d_params = sum(p.numel() for p in model_D.parameters() if p.requires_grad)
    print(f"Generator parameters: {g_params:,}")
    print(f"Discriminator parameters: {d_params:,}")
    
    # Loss functions
    criterion_seg = CombinedGeneratorLoss(
        weight_ce=cfg.LOSS_WEIGHT_CE,
        weight_dice=cfg.LOSS_WEIGHT_DICE,
        weight_mscr=cfg.LOSS_WEIGHT_MSCR
    ).to(cfg.DEVICE)
    
    criterion_adv = AdversarialLoss().to(cfg.DEVICE)
    
    # Optimizers
    opt_G = torch.optim.Adam(
        model_G.parameters(), lr=cfg.LR_GENERATOR, betas=(cfg.B1, cfg.B2)
    )
    opt_D = torch.optim.Adam(
        model_D.parameters(), lr=cfg.LR_DISCRIMINATOR, betas=(cfg.B1, cfg.B2)
    )
    
    # Schedulers
    scheduler_G = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt_G, mode='max', factor=0.5, patience=3, verbose=True
    )
    scheduler_D = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt_D, mode='min', factor=0.5, patience=3, verbose=True
    )
    
    # Mixed precision scalers
    scaler_G = GradScaler() if cfg.USE_MIXED_PRECISION else None
    scaler_D = GradScaler() if cfg.USE_MIXED_PRECISION else None
    
    # Metrics plotter
    plotter = MetricsPlotter()
    
    # Training history
    history = []
    
    # Best model tracking
    best_val_mscr = 0.0
    
    print("\n" + "=" * 80)
    print("Starting Training...")
    print("=" * 80)
    
    for epoch in range(cfg.NUM_EPOCHS):
        start_time = time.time()
        
        # Train
        train_metrics_all = train_one_epoch(
            model_G, model_D, train_loader,
            opt_G, opt_D,
            criterion_seg, criterion_adv,
            scaler_G, scaler_D,
            cfg.DEVICE, epoch
        )
        
        # Validate
        val_metrics = validate(
            model_G, val_loader, criterion_seg, cfg.DEVICE, epoch
        )
        
        epoch_time = time.time() - start_time
        
        # Extract metrics
        train_metrics = train_metrics_all[:13]
        (avg_loss_G_total, avg_loss_G_adv, avg_loss_D_total) = train_metrics_all[13:]
        
        (train_loss, train_ce, train_dice, train_mscr_l,
         train_acc, train_iou, train_iou_land, train_iou_water,
         train_mscr, train_scr_k4, train_scr_k8, train_scr_k24, train_scr_k48) = train_metrics
        
        (val_loss, val_ce, val_dice, val_mscr_l,
         val_acc, val_iou, val_iou_land, val_iou_water,
         val_mscr, val_scr_k4, val_scr_k8, val_scr_k24, val_scr_k48) = val_metrics
        
        # Update schedulers
        scheduler_G.step(val_mscr)
        scheduler_D.step(avg_loss_D_total)
        
        # Print epoch summary
        print("\n" + "=" * 80)
        print(f"EPOCH {epoch + 1}/{cfg.NUM_EPOCHS} Summary (Time: {epoch_time:.2f}s)")
        print("=" * 80)
        print(f"GAN LOSSES:")
        print(f"  Generator (Total): {avg_loss_G_total:.4f}")
        print(f"  Generator (Seg):   {train_loss:.4f}")
        print(f"  Generator (Adv):   {avg_loss_G_adv:.4f}")
        print(f"  Discriminator:     {avg_loss_D_total:.4f}")
        print(f"\nSEGMENTATION LOSSES:")
        print(f"  Total Seg - Train: {train_loss:.4f} | Val: {val_loss:.4f}")
        print(f"  CE        - Train: {train_ce:.4f} | Val: {val_ce:.4f}")
        print(f"  Dice      - Train: {train_dice:.4f} | Val: {val_dice:.4f}")
        print(f"  mSCR Loss - Train: {train_mscr_l:.4f} | Val: {val_mscr_l:.4f}")
        print(f"\nMETRICS:")
        print(f"  Accuracy - Train: {train_acc:.4f} | Val: {val_acc:.4f}")
        print(f"  IoU      - Train: {train_iou:.4f} | Val: {val_iou:.4f}")
        print(f"  mSCR     - Train: {train_mscr:.4f} | Val: {val_mscr:.4f} {'*** NEW BEST ***' if val_mscr > best_val_mscr else ''}")
        print(f"\nSCR by K-values (Validation):")
        print(f"  k=4:  {val_scr_k4:.4f}")
        print(f"  k=8:  {val_scr_k8:.4f}")
        print(f"  k=24: {val_scr_k24:.4f}")
        print(f"  k=48: {val_scr_k48:.4f}")

        # Save to history
        history.append({
            'epoch': epoch + 1,
            'train_loss_seg': train_loss,
            'train_loss_g_total': avg_loss_G_total,
            'train_loss_g_adv': avg_loss_G_adv,
            'train_loss_d': avg_loss_D_total,
            'train_ce': train_ce,
            'train_dice': train_dice,
            'train_mscr_loss': train_mscr_l,
            'train_acc': train_acc,
            'train_iou': train_iou,
            'train_mscr': train_mscr,
            'val_loss': val_loss,
            'val_ce': val_ce,
            'val_dice': val_dice,
            'val_mscr_loss': val_mscr_l,
            'val_acc': val_acc,
            'val_iou': val_iou,
            'val_mscr': val_mscr,
            'val_scr_k4': val_scr_k4,
            'val_scr_k8': val_scr_k8,
            'val_scr_k24': val_scr_k24,
            'val_scr_k48': val_scr_k48,
            'epoch_time': epoch_time
        })
        
        # Save incremental history after each epoch
        history_df = pd.DataFrame(history)
        history_csv_path = os.path.join(cfg.OUTPUT_DIR, 'training_history_gan.csv')
        history_df.to_csv(history_csv_path, index=False)
        
        # Update live plots
        plotter.update(train_metrics, val_metrics)
        
        # Save best model
        if val_mscr > best_val_mscr:
            best_val_mscr = val_mscr
            
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model_G.state_dict(),
                'optimizer_state_dict': opt_G.state_dict(),
                'val_mscr': val_mscr,
                'val_iou': val_iou,
                'config': {
                    'generator_encoder': cfg.GENERATOR_ENCODER,
                    'image_size': cfg.IMAGE_SIZE,
                    'k_neighbors': cfg.K_NEIGHBORS
                }
            }
            
            torch.save(checkpoint, cfg.BEST_MODEL_PATH)
            print(f"\n✓ New best model (Generator) saved! Val mSCR: {val_mscr:.4f}")
        
        print("=" * 80 + "\n")

    # Training complete
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    print(f"Best Validation mSCR: {best_val_mscr:.4f}")
    print(f"Best Generator saved at: {cfg.BEST_MODEL_PATH}")
    
    # Save final training curves
    final_plot_path = os.path.join(cfg.OUTPUT_DIR, 'training_curves_gan.png')
    plotter.save_final_plot(final_plot_path)
    
    # ============================================================================
    # FINAL TEST SET EVALUATION
    # ============================================================================
    print("\n" + "=" * 80)
    print("FINAL EVALUATION ON TEST SET")
    print("=" * 80)
    
    # Load test dataset
    test_dataset = CAIDDataset(
        split_file=cfg.TEST_TXT,
        images_dir=cfg.IMAGES_DIR,
        masks_dir=cfg.MASKS_DIR,
        augment=False
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False,
        num_workers=cfg.NUM_WORKERS, pin_memory=cfg.PIN_MEMORY
    )
    
    print(f"Test batches: {len(test_loader)}")
    
    # Load best model
    print(f"\nLoading best model from: {cfg.BEST_MODEL_PATH}")
    checkpoint = torch.load(cfg.BEST_MODEL_PATH)
    model_G.load_state_dict(checkpoint['model_state_dict'])
    print(f"Best model was from epoch {checkpoint['epoch']} with Val mSCR: {checkpoint['val_mscr']:.4f}")
    
    # Evaluate on test set
    test_metrics = validate(
        model_G, test_loader, criterion_seg, cfg.DEVICE, epoch=cfg.NUM_EPOCHS
    )
    
    (test_loss, test_ce, test_dice, test_mscr_l,
     test_acc, test_iou, test_iou_land, test_iou_water,
     test_mscr, test_scr_k4, test_scr_k8, test_scr_k24, test_scr_k48) = test_metrics
    
    # Print final test results
    print("\n" + "=" * 80)
    print("FINAL TEST SET RESULTS")
    print("=" * 80)
    print(f"Test Loss (Seg): {test_loss:.4f}")
    print(f"  CE Loss:   {test_ce:.4f}")
    print(f"  Dice Loss: {test_dice:.4f}")
    print(f"  mSCR Loss: {test_mscr_l:.4f}")
    print(f"\nTest Metrics:")
    print(f"  Accuracy:  {test_acc:.4f}")
    print(f"  Mean IoU:  {test_iou:.4f}")
    print(f"    - Land IoU:  {test_iou_land:.4f}")
    print(f"    - Water IoU: {test_iou_water:.4f}")
    print(f"  mSCR:      {test_mscr:.4f}")
    print(f"\nSCR by K-values:")
    print(f"  k=4:  {test_scr_k4:.4f}")
    print(f"  k=8:  {test_scr_k8:.4f}")
    print(f"  k=24: {test_scr_k24:.4f}")
    print(f"  k=48: {test_scr_k48:.4f}")
    print("=" * 80)
    
    # Save test results
    test_results = {
        'test_loss': test_loss,
        'test_ce': test_ce,
        'test_dice': test_dice,
        'test_mscr_loss': test_mscr_l,
        'test_acc': test_acc,
        'test_iou': test_iou,
        'test_iou_land': test_iou_land,
        'test_iou_water': test_iou_water,
        'test_mscr': test_mscr,
        'test_scr_k4': test_scr_k4,
        'test_scr_k8': test_scr_k8,
        'test_scr_k24': test_scr_k24,
        'test_scr_k48': test_scr_k48,
        'best_val_epoch': checkpoint['epoch'],
        'best_val_mscr': checkpoint['val_mscr']
    }
    
    test_results_df = pd.DataFrame([test_results])
    test_results_path = os.path.join(cfg.OUTPUT_DIR, 'test_results_gan.csv')
    test_results_df.to_csv(test_results_path, index=False)
    print(f"\n✓ Test results saved to: {test_results_path}")
    
    print("\n✓ All done! Check the output directory for results.")
    
    return model_G, history, test_results

# ============================================================================
# SCRIPT ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    # Run training
    model, history, test_results = main()
    
    print("\n" + "=" * 80)
    print("TRAINING AND EVALUATION COMPLETE!")
    print("=" * 80)
    print("\nGenerated Files:")
    print(f"  1. Best Generator Model: {cfg.BEST_MODEL_PATH}")
    print(f"  2. Training History CSV: {os.path.join(cfg.OUTPUT_DIR, 'training_history_gan.csv')}")
    print(f"  3. Training Curves Plot: {os.path.join(cfg.OUTPUT_DIR, 'training_curves_gan.png')}")
    print(f"  4. Test Results CSV: {os.path.join(cfg.OUTPUT_DIR, 'test_results_gan.csv')}")
    print("\nKey Results:")
    print(f"  Best Val mSCR: {history[-1]['val_mscr']:.4f}")
    print(f"  Test mSCR: {test_results['test_mscr']:.4f}")
    print(f"  Test IoU: {test_results['test_iou']:.4f}")
    print(f"  Test Accuracy: {test_results['test_acc']:.4f}")
    print("=" * 80)