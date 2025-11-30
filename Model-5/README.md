# Model 5: HRNet-W48 with Hybrid Edge Channels - Extended Implementation

## Overview

This implementation presents an enhanced version of the High-Resolution Network (HRNet-W48) with hybrid edge channel inputs for semantic segmentation of the CAID (Coastal Aerial Imagery Dataset). The model incorporates explicit edge information through Sobel gradient magnitude and Canny edge detection channels, combined with comprehensive logging and evaluation infrastructure.

## Architecture

### HRNet-W48 Backbone
- **Network Type**: High-Resolution Network with width-48 configuration
- **Key Design**: Maintains high-resolution representations throughout the network depth
- **Parallel Branches**: Four parallel branches at different resolutions (full, 1/2, 1/4, 1/8)
- **Multi-Resolution Fusion**: Repeated exchange and fusion across resolution branches
- **Feature Channels**: 48, 96, 192, 384 channels for respective branches

### Hybrid Input Architecture
The model processes 5-channel inputs:
1. **Red Channel**: First color component
2. **Green Channel**: Second color component
3. **Blue Channel**: Third color component
4. **Sobel Magnitude**: Gradient-based edge strength
5. **Canny Edges**: Binary edge map

### Edge Detection Pipeline

#### Sobel Magnitude Computation
- Convert RGB to grayscale using OpenCV
- Apply Sobel operator in X and Y directions (3×3 kernel)
- Compute magnitude: sqrt(Sobel_x² + Sobel_y²)
- Normalize to [0, 1] range

#### Canny Edge Detection
- Convert RGB to grayscale
- Apply Canny algorithm with thresholds (50, 150)
- Output binary edge map normalized to [0, 1]

### Segmentation Head
- Multi-resolution feature aggregation from HRNet branches
- Convolution layers with batch normalization
- Dropout regularization (0.1)
- Final 1×1 convolution for class prediction
- Bilinear upsampling to original input resolution

## Training Configuration

### Hyperparameters
- **Batch Size**: 8 (optimized for HRNet-W48 memory footprint)
- **Training Epochs**: 50
- **Base Learning Rate**: 1e-4
- **Weight Decay**: 1e-4
- **Optimizer**: Adam with default betas (0.9, 0.999)
- **Input Resolution**: 500 × 500 pixels

### Loss Function Components
Multi-objective loss function:
- **Cross-Entropy Loss** (weight: 1.0): Standard pixel-wise classification loss
- **Dice Loss** (weight: 1.0): Overlap-based region similarity
- **Differentiable mSCR Loss** (weight: 0.5): Shoreline conformity optimization

### Data Augmentation Strategy
- Random horizontal flip (probability: 0.5)
- Random vertical flip (probability: 0.5)
- Random rotation at 90°, 180°, or 270° (probability: 0.3)
- Edge channel recomputation after geometric transformations
- ImageNet normalization for RGB channels only

## Evaluation Framework

### Mean Shoreline Conformity Rate (mSCR)
- **Definition**: Average shoreline conformity across multiple neighborhood sizes
- **Neighborhood Sizes**: k ∈ {4, 8, 24, 48} pixels
- **Computation**: Per-k averaging methodology
- **Bidirectional**: Evaluates both prediction→ground-truth and ground-truth→prediction

#### SCR Computation Formula
```
SCR_k = (2 × |S1 ∩ S(S2, N)|) / (|S1| + |S2|)
where:
  S1 = predicted shoreline
  S2 = ground truth shoreline
  S(S2, N) = N-neighborhood expansion of S2
  N = neighborhood size k
```

### Additional Evaluation Metrics
- **Intersection over Union (IoU)**: Per-class and mean IoU
- **Pixel Accuracy**: Overall correct classification rate
- **Precision**: True positive / (True positive + False positive)
- **Recall**: True positive / (True positive + False negative)
- **Per-k SCR**: Individual SCR values for each neighborhood size

## Dataset Specifications

### CAID Dataset Organization
- **Dataset Root**: `/kaggle/input/caid-dataset/CAID`
- **Image Directory**: `JPEGImages/` (PNG format, RGB)
- **Mask Directory**: `SegmentationClass/` (PNG format, grayscale)
- **Train Split**: `ImageSets/Segmentation/train.txt`
- **Validation Split**: `ImageSets/Segmentation/val.txt`
- **Test Split**: `ImageSets/Segmentation/test.txt`
- **Label Encoding**: 0 for land, 1 for water

### Data Pipeline
1. Load image ID from split file
2. Read RGB image from JPEGImages directory
3. Read binary mask from SegmentationClass directory
4. Compute Sobel magnitude and Canny edges
5. Concatenate to form 5-channel input
6. Apply synchronized augmentations
7. Normalize and convert to tensors

## Implementation Details

### Logging Infrastructure
Comprehensive logging system with:
- **Logger Name**: HRNet_CAID
- **File Handler**: DEBUG level, timestamped log files
- **Console Handler**: INFO level for real-time monitoring
- **Log Directory**: `logs/` subdirectory in output path
- **Format**: Timestamp | Logger | Level | Message
- **Metrics Tracked**: Losses, mSCR, IoU, accuracy per epoch

### Mixed Precision Training
- **Framework**: PyTorch Automatic Mixed Precision (AMP)
- **GradScaler**: Dynamic loss scaling to prevent underflow
- **Benefits**: Reduced memory usage, faster training
- **Precision**: FP16 for forward/backward, FP32 for weights

### Differentiable mSCR Loss Implementation
- **Shoreline Extraction**: Laplacian operator with 3×3 kernel
- **Edge Detection Threshold**: Absolute value > 0.1
- **Neighborhood Expansion**: Max pooling with adaptive kernel sizes
  - k=4 → 3×3 kernel
  - k=8 → 3×3 kernel
  - k=24 → 5×5 kernel
  - k=48 → 7×7 kernel
- **Differentiability**: Fully differentiable for gradient-based optimization

### Model Checkpointing
- **Criterion**: Best validation mSCR
- **Saved Components**: Model state dictionary
- **Path**: `hrnet_hybrid_best.pth` in output directory
- **Format**: PyTorch state dictionary

## Computational Requirements

### Hardware
- **GPU**: CUDA-enabled GPU with adequate VRAM
- **Recommended Memory**: Minimum 16GB GPU memory for batch size 8
- **CPU**: Multi-core processor for data loading
- **Storage**: Sufficient space for dataset and checkpoints

### Software Dependencies
- **PyTorch**: Deep learning framework with CUDA support
- **OpenCV (cv2)**: Image processing and edge detection
- **NumPy**: Numerical computations
- **PIL (Pillow)**: Image loading
- **Matplotlib**: Visualization (Agg backend)
- **scipy**: Binary morphological operations
- **logging**: Comprehensive logging system

## Usage Instructions

### Training Execution
1. Ensure CAID dataset is available at specified path
2. Configure hyperparameters in Config class if needed
3. Execute training script
4. Monitor console output for per-epoch metrics
5. Check logs directory for detailed training logs
6. Best model automatically saved to output directory

### Model Inference
1. Load trained checkpoint: `torch.load('hrnet_hybrid_best.pth')`
2. Initialize HRNet-W48 with 5-channel input
3. Preprocess input: create hybrid 5-channel tensor
4. Forward pass through model
5. Apply softmax and threshold for binary mask
6. Post-process as needed

## Key Technical Features

1. **High-Resolution Maintenance**: Preserves spatial details throughout network
2. **Multi-Scale Feature Fusion**: Repeated information exchange across resolutions
3. **Explicit Edge Information**: Sobel and Canny channels guide boundary learning
4. **Differentiable Boundary Loss**: End-to-end optimization for shoreline accuracy
5. **Comprehensive Logging**: Detailed tracking for debugging and analysis
6. **Per-k SCR Analysis**: Fine-grained boundary evaluation at multiple scales

## Model Advantages

- **Superior Boundary Localization**: Edge channels explicitly encode boundaries
- **High Spatial Precision**: HRNet maintains resolution throughout
- **Multi-Scale Context**: Parallel branches capture features at different scales
- **Robust Training**: Mixed precision and comprehensive augmentation
- **Interpretable Metrics**: Per-k SCR provides scale-specific boundary analysis

## Performance Monitoring

The implementation provides:
- Real-time console output during training
- Detailed log files with timestamp information
- Per-epoch metrics (loss components, mSCR, IoU, accuracy)
- Best model checkpoint based on validation mSCR
- Comprehensive test evaluation with per-k SCR breakdown

## Research Considerations

This implementation enables investigation of:
- Impact of edge channels on boundary accuracy
- HRNet effectiveness for coastal segmentation
- Per-k SCR analysis for different shoreline scales
- Loss weight sensitivity analysis
- Augmentation strategy effectiveness

## Reproducibility

For reproducible results:
- Set random seeds for PyTorch, NumPy, and Python
- Use deterministic algorithms where available
- Document all hyperparameters in Config class
- Save comprehensive logs for each training run
- Version control dataset splits and preprocessing
