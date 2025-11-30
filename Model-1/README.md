# Model 1: Context Encoding Network (EncNet)

## Overview

This implementation presents a Context Encoding Network (EncNet) for semantic segmentation of coastal aerial imagery. The model performs binary classification to distinguish land from water regions in the CAID (Coastal Aerial Imagery Dataset).

## Architecture

### Backbone
- **Base Architecture**: ResNet-101 pretrained on ImageNet
- **Feature Extraction**: Four-stage hierarchical feature extraction with progressively increasing receptive fields
- **Output Channels**: 2048-dimensional feature maps from the final layer

### Context Encoding Module
The model incorporates a novel Context Encoding Module that learns semantic context through:
- **Learnable Codewords**: 32 learnable visual codewords that capture semantic patterns
- **Soft Assignment**: Soft assignment mechanism using L2 distance-based similarity
- **Residual Encoding**: Weighted aggregation of residuals between features and codewords
- **Channel Attention**: Squeeze-and-Excitation (SE) block for adaptive channel recalibration

### Segmentation Head
- Convolutional layers with batch normalization
- Dropout regularization (0.1) to prevent overfitting
- Bilinear upsampling to restore original input resolution

## Training Configuration

### Hyperparameters
- **Batch Size**: 16
- **Epochs**: 50
- **Learning Rate**: 1e-4
- **Weight Decay**: 1e-4
- **Optimizer**: Adam
- **Image Size**: 500 × 500 pixels

### Loss Function
Multi-objective loss combining:
- **Cross-Entropy Loss** (weight: 1.0): Pixel-wise classification
- **Dice Loss** (weight: 1.0): Region-based overlap optimization
- **Differentiable mSCR Loss** (weight: 0.5): Shoreline conformity optimization

### Data Augmentation
- Horizontal and vertical flipping (probability: 0.5)
- Random rotation at 90°, 180°, 270° (probability: 0.3)
- ImageNet normalization (mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225])

## Evaluation Metrics

### Mean Shoreline Conformity Rate (mSCR)
The primary evaluation metric assesses boundary accuracy across multiple neighborhood sizes:
- **Neighborhood Sizes**: k ∈ {4, 8, 24, 48}
- **Bidirectional Evaluation**: Measures conformity from predicted to ground truth and vice versa
- **Formula**: mSCR = average of SCR_k across all k values

### Additional Metrics
- **Intersection over Union (IoU)**: Per-class and mean IoU
- **Pixel Accuracy**: Overall classification accuracy
- **Precision and Recall**: Class-wise performance metrics

## Dataset

### CAID Dataset Structure
- **Training Set**: Images specified in train.txt
- **Validation Set**: Images specified in val.txt
- **Test Set**: Images specified in test.txt
- **Format**: PNG images with corresponding segmentation masks
- **Classes**: Binary (0: Land, 1: Water)

### Data Pipeline
- Synchronized augmentation for images and masks
- On-the-fly preprocessing and normalization
- Multi-threaded data loading with 2 workers

## Implementation Details

### Mixed Precision Training
- Automatic Mixed Precision (AMP) enabled for faster training
- GradScaler for gradient scaling to prevent underflow

### Differentiable mSCR Loss
- Utilizes Laplacian edge detection for shoreline extraction
- Max pooling-based neighborhood expansion
- Soft assignment for gradient flow during backpropagation

### Computational Requirements
- **Device**: CUDA-enabled GPU
- **Memory**: Sufficient VRAM for batch size 16 with ResNet-101
- **Framework**: PyTorch with torchvision

## Usage

### Training
The model automatically trains for 50 epochs with early stopping based on validation mSCR. The best model checkpoint is saved to `encnet_best.pth`.

### Inference
The trained model outputs:
- Segmentation logits of shape (B, 2, H, W)
- Context-encoded feature vectors for semantic understanding

## Key Features

1. **Context-Aware Segmentation**: Learns global semantic context through learnable codewords
2. **Multi-Scale Loss**: Combines pixel-level, region-level, and boundary-level objectives
3. **Differentiable Evaluation**: mSCR loss enables end-to-end optimization for boundary accuracy
4. **Robust Augmentation**: Comprehensive data augmentation strategy for improved generalization

## Results

The model is evaluated on:
- Training set performance (logged per epoch)
- Validation set performance (for model selection)
- Test set performance (final evaluation)

Performance metrics include mSCR, IoU, pixel accuracy, precision, and recall across both land and water classes.
