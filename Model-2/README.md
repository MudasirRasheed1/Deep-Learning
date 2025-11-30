# Model 2: Context Encoding Network (EncNet) - Configuration Variant

## Overview

This implementation presents a Context Encoding Network (EncNet) for semantic segmentation of the CAID (Coastal Aerial Imagery Dataset). The model performs binary classification between land and water regions using a ResNet-101 backbone with a context encoding module.

## Architecture

### Backbone
- **Base Architecture**: ResNet-101 pretrained on ImageNet
- **Feature Hierarchy**: Four-stage feature extraction (256, 512, 1024, 2048 channels)
- **Receptive Field**: Progressively increasing spatial context

### Context Encoding Module
The model employs a Context Encoding Module consisting of:
- **Context Encoding Layer**: 32 learnable codewords for semantic pattern capture
- **Soft Assignment**: Distance-based soft assignment to codewords
- **Residual Aggregation**: Weighted sum of feature-codeword residuals
- **Squeeze-and-Excitation**: Channel attention mechanism for feature recalibration

### Segmentation Head
- 3×3 convolution with batch normalization
- Dropout regularization (0.1 probability)
- 1×1 convolution for class prediction
- Bilinear upsampling to input resolution

## Training Configuration

### Hyperparameters
- **Batch Size**: 16
- **Total Epochs**: 50
- **Learning Rate**: 1e-4
- **Weight Decay**: 1e-4
- **Optimizer**: Adam
- **Input Resolution**: 500 × 500 pixels

### Loss Function
Composite loss function with three components:
- **Cross-Entropy Loss** (weight: 1.0): Standard pixel-wise classification
- **Dice Loss** (weight: 1.0): Overlap-based region optimization
- **Differentiable mSCR Loss** (weight: 0.5): Boundary conformity optimization

### Data Augmentation Strategy
- Random horizontal flip (probability: 0.5)
- Random vertical flip (probability: 0.5)
- Random rotation: 90°, 180°, or 270° (probability: 0.3)
- ImageNet-based normalization

## Evaluation Framework

### Primary Metric: Mean Shoreline Conformity Rate (mSCR)
- **Neighborhood Sizes**: k ∈ {4, 8, 24, 48} pixels
- **Bidirectional Assessment**: Evaluates both prediction-to-ground-truth and ground-truth-to-prediction conformity
- **Aggregation**: Average SCR across all k values

### Secondary Metrics
- **Intersection over Union (IoU)**: Per-class and mean IoU
- **Pixel Accuracy**: Overall correct pixel classification rate
- **Precision and Recall**: Class-specific performance indicators

## Dataset Specifications

### CAID Dataset
- **Data Root**: `/kaggle/input/caid-dataset/CAID`
- **Image Directory**: `JPEGImages/`
- **Mask Directory**: `SegmentationClass/`
- **Splits**: train.txt, val.txt, test.txt
- **Format**: PNG images and corresponding binary masks
- **Label Encoding**: 0 (Land), 1 (Water)

### Data Loading
- Custom PyTorch Dataset class with synchronized augmentation
- Multi-threaded loading with 2 workers
- Pin memory optimization for faster GPU transfer

## Implementation Details

### Differentiable mSCR Loss
- **Shoreline Extraction**: Laplacian-based edge detection using 3×3 kernel
- **Neighborhood Expansion**: Max pooling with kernel sizes {3×3, 3×3, 5×5, 7×7} for k ∈ {4, 8, 24, 48}
- **Gradient Flow**: Fully differentiable for end-to-end optimization

### Mixed Precision Training
- Automatic Mixed Precision (AMP) enabled
- GradScaler for dynamic loss scaling
- Improved training speed and memory efficiency

### Model Checkpointing
- Best model saved based on validation mSCR
- Checkpoint path: `encnet_best.pth`
- Includes model state, optimizer state, and training metrics

## Computational Requirements

- **Platform**: CUDA-enabled GPU
- **Memory**: Adequate VRAM for batch size 16 with ResNet-101 backbone
- **Dependencies**: PyTorch, torchvision, NumPy, PIL, matplotlib, scipy

## Usage Instructions

### Training Process
1. Dataset is automatically loaded from specified paths
2. Model trains for 50 epochs with validation after each epoch
3. Best model is saved based on highest validation mSCR
4. Training progress displayed with loss components and metrics

### Model Output
- **Segmentation Map**: Binary classification logits (B, 2, H, W)
- **Encoded Features**: Global context vectors for semantic analysis

## Key Characteristics

1. **Global Context Modeling**: Learnable codewords capture dataset-specific semantic patterns
2. **Multi-Objective Optimization**: Balances pixel accuracy, region overlap, and boundary precision
3. **Differentiable Boundary Metric**: Enables direct optimization of shoreline conformity
4. **Robust Training**: Comprehensive augmentation and regularization strategies

## Performance Evaluation

The model reports:
- Per-epoch training and validation metrics
- Best validation performance checkpoint
- Final test set evaluation with comprehensive metrics
- Per-neighborhood-size SCR breakdown for detailed boundary analysis
