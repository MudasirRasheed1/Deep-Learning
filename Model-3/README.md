# Model 3: HRNet-W48 with Hybrid Edge Channels

## Overview

This implementation presents a High-Resolution Network (HRNet-W48) enhanced with hybrid edge channel inputs for semantic segmentation of the CAID (Coastal Aerial Imagery Dataset). The model distinguishes between land and water regions while incorporating explicit edge information through Sobel and Canny edge detection channels.

## Architecture

### HRNet-W48 Backbone
- **Architecture**: High-Resolution Network with width-48 configuration
- **Key Feature**: Maintains high-resolution representations throughout the network
- **Parallel Branches**: Four parallel branches processing features at different resolutions
- **Multi-Scale Fusion**: Repeated fusion of multi-resolution representations

### Hybrid Input Configuration
The model accepts 5-channel inputs:
1. **RGB Channels**: Standard color information
2. **Sobel Magnitude Channel**: Gradient-based edge detection
3. **Canny Edge Channel**: Binary edge map from Canny edge detector

### Edge Detection Parameters
- **Sobel Kernel Size**: 3×3
- **Canny Lower Threshold**: 50
- **Canny Upper Threshold**: 150

### Segmentation Head
- High-resolution feature aggregation
- Convolution layers with batch normalization
- Dropout regularization
- Bilinear upsampling to original resolution

## Training Configuration

### Hyperparameters
- **Batch Size**: 8 (reduced for HRNet-W48 memory requirements)
- **Total Epochs**: 50
- **Learning Rate**: 1e-4
- **Weight Decay**: 1e-4
- **Optimizer**: Adam
- **Input Resolution**: 500 × 500 pixels

### Loss Function
Multi-component loss function:
- **Cross-Entropy Loss** (weight: 1.0): Pixel-wise classification
- **Dice Loss** (weight: 1.0): Region overlap optimization
- **Differentiable mSCR Loss** (weight: 0.5): Shoreline boundary optimization

### Data Augmentation
- Horizontal flip (probability: 0.5)
- Vertical flip (probability: 0.5)
- Random rotation: 90°, 180°, 270° (probability: 0.3)
- Edge channels recomputed after augmentation
- ImageNet normalization for RGB channels

## Evaluation Metrics

### Mean Shoreline Conformity Rate (mSCR)
- **Neighborhood Sizes**: k ∈ {4, 8, 24, 48}
- **Per-k Averaging**: Separate computation and averaging for each neighborhood size
- **Bidirectional Evaluation**: Measures conformity in both directions

### Additional Metrics
- **Intersection over Union (IoU)**: Per-class and mean IoU
- **Pixel Accuracy**: Overall classification accuracy
- **Precision and Recall**: Class-wise performance
- **Per-k SCR**: Individual SCR scores for each neighborhood size

## Dataset Specifications

### CAID Dataset Structure
- **Data Root**: `/kaggle/input/caid-dataset/CAID`
- **Images**: `JPEGImages/` (PNG format)
- **Masks**: `SegmentationClass/` (PNG format, grayscale)
- **Splits**: train.txt, val.txt, test.txt
- **Classes**: Binary (0: Land, 1: Water)

### Preprocessing Pipeline
1. Load RGB image
2. Compute Sobel magnitude from grayscale conversion
3. Compute Canny edges from grayscale conversion
4. Concatenate to form 5-channel tensor
5. Normalize RGB channels with ImageNet statistics
6. Apply synchronized augmentations

## Implementation Details

### Edge Detection Implementation

#### Sobel Magnitude Computation
```
1. Convert RGB to grayscale
2. Compute Sobel-X and Sobel-Y gradients
3. Calculate magnitude: sqrt(Sobel_x² + Sobel_y²)
4. Normalize to [0, 1] range
```

#### Canny Edge Computation
```
1. Convert RGB to grayscale
2. Apply Canny edge detector with thresholds (50, 150)
3. Normalize binary output to [0, 1] range
```

### Hybrid Input Creation
The model processes 5-channel inputs where:
- Channels 0-2: Normalized RGB values
- Channel 3: Normalized Sobel magnitude
- Channel 4: Binary Canny edges

### Mixed Precision Training
- Automatic Mixed Precision (AMP) enabled
- GradScaler for dynamic loss scaling
- Improved memory efficiency and training speed

### Logging System
Comprehensive logging infrastructure:
- **Console Handler**: INFO level messages
- **File Handler**: DEBUG level detailed logs
- **Log Files**: Timestamped training logs in `logs/` directory
- **Metrics Tracking**: Per-epoch loss components and evaluation metrics

## Computational Requirements

- **Device**: CUDA-enabled GPU with sufficient VRAM
- **Memory**: Higher requirements due to HRNet-W48 architecture and 5-channel input
- **Dependencies**: PyTorch, OpenCV, NumPy, PIL, matplotlib, scipy, logging

## Usage

### Training Workflow
1. Load CAID dataset with 5-channel preprocessing
2. Initialize HRNet-W48 with modified first convolution for 5 channels
3. Train for 50 epochs with validation after each epoch
4. Save best model based on validation mSCR
5. Generate comprehensive training logs

### Model Output
- **Segmentation Logits**: (B, 2, H, W) classification scores
- **Edge-Enhanced Features**: Improved boundary localization

## Key Innovations

1. **Hybrid Edge Channels**: Explicit edge information augments RGB input
2. **High-Resolution Processing**: HRNet maintains spatial resolution throughout
3. **Multi-Scale Fusion**: Repeated exchange of information across resolutions
4. **Enhanced Boundary Detection**: Edge channels improve shoreline localization
5. **Comprehensive Logging**: Detailed tracking of training progress and metrics

## Model Advantages

- **Improved Boundary Accuracy**: Edge channels provide explicit boundary cues
- **High-Resolution Representations**: Better spatial precision for segmentation
- **Multi-Scale Feature Fusion**: Captures both fine details and global context
- **Robust to Scale Variations**: Parallel multi-resolution processing

## Performance Analysis

The model provides:
- Per-epoch training and validation metrics
- Per-k SCR breakdown for detailed boundary analysis
- Best model checkpoint based on validation mSCR
- Comprehensive test set evaluation
- Detailed logs for training diagnostics

## Ablation Considerations

The hybrid edge channel approach can be evaluated by:
- Comparing 5-channel vs. 3-channel (RGB-only) input
- Analyzing per-k SCR improvements
- Evaluating computational overhead of edge detection
- Assessing boundary localization improvements
