# Model 6: HRNet-W48 with Hybrid Edge Channels - Final Version v2

## Overview

This implementation represents the final optimized version (v2) of the High-Resolution Network (HRNet-W48) with hybrid edge channel inputs for semantic segmentation of the CAID (Coastal Aerial Imagery Dataset). This version incorporates critical bug fixes, upgraded loss functions, improved optimizer configuration, and enhanced post-processing for superior boundary accuracy in land-water classification.

## Version History and Improvements

### Key Modifications in v2
1. **Bug Fix**: Corrected Sobel magnitude calculation (sobelx**2 instead of sobelx*2)
2. **Metric Fix**: Reverted SCR calculation to original formula for fair comparison
3. **Loss Upgrade 1**: Replaced Cross-Entropy with Focal Loss for class imbalance handling
4. **Loss Upgrade 2**: Replaced Dice Loss with Lovasz-Softmax Loss for better boundary optimization
5. **Optimizer Upgrade**: Replaced Adam with AdamW for improved weight decay regularization
6. **Scheduler Upgrade**: Replaced ReduceLROnPlateau with CosineAnnealingLR for smoother convergence
7. **Post-Processing**: Lowered minimum area threshold to 25 pixels for finer detail preservation

## Architecture

### HRNet-W48 Backbone
- **Architecture Type**: High-Resolution Network with width-48 configuration
- **Core Design Philosophy**: Maintains high-resolution representations throughout network depth
- **Multi-Resolution Branches**: Four parallel streams at resolutions (1, 1/2, 1/4, 1/8)
- **Channel Configuration**: [48, 96, 192, 384] channels across branches
- **Fusion Strategy**: Repeated multi-scale fusion with strided convolutions and upsampling

### Hybrid 5-Channel Input
The model processes augmented input with explicit edge information:
1. **Channel 0 (Red)**: First RGB component, ImageNet normalized
2. **Channel 1 (Green)**: Second RGB component, ImageNet normalized
3. **Channel 2 (Blue)**: Third RGB component, ImageNet normalized
4. **Channel 3 (Sobel)**: Gradient magnitude from Sobel operator, normalized [0,1]
5. **Channel 4 (Canny)**: Binary edge map from Canny detector, normalized [0,1]

### Edge Detection Implementation

#### Sobel Magnitude (Bug Fixed)
```python
# Correct implementation
sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
magnitude = np.sqrt(sobelx**2 + sobely**2)  # Fixed: was sobelx*2
magnitude = magnitude / magnitude.max()  # Normalize
```

#### Canny Edge Detection
- **Threshold 1**: 50 (lower threshold for edge linking)
- **Threshold 2**: 150 (upper threshold for strong edges)
- **Output**: Binary edge map [0, 1]

### Segmentation Architecture
- Multi-resolution feature aggregation from all HRNet branches
- Convolution with batch normalization
- Dropout regularization (probability: 0.1)
- Final 1×1 convolution for binary classification
- Bilinear interpolation to input resolution

### Auxiliary Boundary Head
- **Purpose**: Explicit boundary supervision
- **Architecture**: Separate decoder branch for boundary prediction
- **Loss Weight**: 0.3 (balanced with segmentation objectives)

## Training Configuration

### Hyperparameters
- **Batch Size**: 8 (memory-optimized for HRNet-W48)
- **Total Epochs**: 50
- **Initial Learning Rate**: 1e-4
- **Weight Decay**: 1e-2 (true weight decay for AdamW)
- **Input Resolution**: 500 × 500 pixels

### Optimizer: AdamW
- **Type**: Adam with decoupled weight decay
- **Betas**: (0.9, 0.999) - default Adam momentum parameters
- **Epsilon**: 1e-8 - numerical stability
- **Weight Decay**: 1e-2 - properly decoupled L2 regularization
- **Advantages**: Better generalization, more effective regularization

### Learning Rate Scheduler: CosineAnnealingLR
- **Type**: Cosine annealing without warm restarts
- **T_max**: 50 epochs (full training duration)
- **Eta_min**: 0 (minimum learning rate)
- **Schedule**: Smooth cosine decay from initial LR to 0
- **Advantages**: No manual threshold tuning, smooth convergence

### Loss Function (Upgraded)

#### Focal Loss (Replaces Cross-Entropy)
- **Purpose**: Addresses class imbalance by down-weighting easy examples
- **Formula**: FL(p_t) = -α(1-p_t)^γ log(p_t)
- **Gamma**: 2.0 (focusing parameter)
- **Alpha**: Class-balanced weights
- **Weight**: 1.0
- **Advantage**: Better handling of land-water imbalance

#### Lovasz-Softmax Loss (Replaces Dice)
- **Purpose**: Direct optimization of IoU metric
- **Method**: Lovasz extension of Jaccard loss
- **Properties**: Convex surrogate for IoU
- **Weight**: 1.0
- **Advantage**: Better boundary optimization than Dice

#### Differentiable mSCR Loss
- **Purpose**: Shoreline conformity optimization
- **Method**: Differentiable approximation of SCR metric
- **Weight**: 0.5
- **Components**: Laplacian edge detection + max pooling expansion

#### Boundary Loss
- **Purpose**: Explicit boundary supervision
- **Method**: Binary cross-entropy on boundary predictions
- **Weight**: 0.3
- **Target**: Ground truth boundaries extracted via morphological operations

### Total Loss Function
```
Total Loss = 1.0 × Focal Loss
           + 1.0 × Lovasz-Softmax Loss
           + 0.5 × mSCR Loss
           + 0.3 × Boundary Loss
```

### Data Augmentation
- Random horizontal flip (probability: 0.5)
- Random vertical flip (probability: 0.5)
- Random rotation: 90°, 180°, 270° (probability: 0.3)
- Edge channel recomputation after geometric transforms
- ImageNet normalization for RGB channels

## Evaluation Metrics

### Mean Shoreline Conformity Rate (mSCR)
- **Neighborhood Sizes**: k ∈ {4, 8, 24, 48} pixels
- **Formula**: mSCR = mean(SCR_4, SCR_8, SCR_24, SCR_48)
- **Per-k Computation**: Average of bidirectional SCR
- **Bidirectionality**: Both pred→GT and GT→pred conformity

### Additional Metrics
- **Intersection over Union (IoU)**: Per-class and mean IoU
- **Pixel Accuracy**: Overall classification accuracy
- **Precision**: Class-wise precision scores
- **Recall**: Class-wise recall scores
- **F1-Score**: Harmonic mean of precision and recall

## Post-Processing (Enhanced)

### Morphological Filtering
- **Enabled**: Configurable via `USE_POST_PROCESSING` flag
- **Method**: Connected component analysis
- **Minimum Area Threshold**: 25 pixels (lowered from higher values)
- **Rationale**: Removes only tiny speckles while preserving fine details
- **Operation**: Remove connected components with area < 25 pixels

### Threshold Selection
The lowered threshold (25 pixels) provides:
- Preservation of narrow coastal features
- Removal of noise-induced artifacts
- Balance between detail retention and noise suppression

## Dataset Specifications

### CAID Dataset Structure
- **Root Directory**: `/kaggle/input/caid-dataset/CAID`
- **Image Source**: `JPEGImages/` (PNG format, RGB)
- **Mask Source**: `SegmentationClass/` (PNG format, binary)
- **Train Split**: `ImageSets/Segmentation/train.txt`
- **Validation Split**: `ImageSets/Segmentation/val.txt`
- **Test Split**: `ImageSets/Segmentation/test.txt`
- **Classes**: 0 (land), 1 (water)

### Preprocessing Pipeline
1. Load RGB image and binary mask
2. Compute Sobel magnitude (corrected formula)
3. Compute Canny edge map
4. Concatenate to 5-channel tensor
5. Apply synchronized augmentations
6. Normalize RGB channels (ImageNet statistics)
7. Convert to PyTorch tensors

## Implementation Details

### Logging System
- **Logger**: HRNet_CAID with dual handlers
- **File Logging**: DEBUG level, timestamped files in `logs/`
- **Console Logging**: INFO level for monitoring
- **Duplicate Prevention**: Checks for existing handlers before adding
- **Format**: `Timestamp | Logger | Level | Message`

### Mixed Precision Training
- **Framework**: PyTorch AMP (Automatic Mixed Precision)
- **Scaler**: GradScaler for dynamic loss scaling
- **Precision**: FP16 operations with FP32 master weights
- **Benefits**: 2x speedup, 40% memory reduction

### Differentiable mSCR Loss
- **Shoreline Extraction**: Laplacian kernel [-8 center, 1 neighbors]
- **Threshold**: |response| > 0.1 for binary shoreline
- **Expansion**: Max pooling with adaptive kernel sizes
  - k=4 → 3×3 kernel (1-pixel expansion)
  - k=8 → 3×3 kernel (1-pixel expansion)
  - k=24 → 5×5 kernel (2-pixel expansion)
  - k=48 → 7×7 kernel (3-pixel expansion)
- **Differentiability**: End-to-end gradient flow

### Model Checkpointing
- **Selection Criterion**: Best validation mSCR
- **Save Path**: `hrnet_hybrid_best.pth`
- **Contents**: Model state dictionary only
- **Format**: PyTorch serialized state

## Computational Requirements

### Hardware Specifications
- **GPU**: CUDA-enabled GPU with minimum 16GB VRAM
- **Recommended**: NVIDIA RTX 3090, A100, or equivalent
- **CPU**: Multi-core processor for data preprocessing
- **RAM**: Minimum 32GB system memory
- **Storage**: ~50GB for dataset and checkpoints

### Software Dependencies
- **PyTorch**: ≥1.10 with CUDA support
- **OpenCV**: Image processing and edge detection
- **NumPy**: <2.0 (compatibility requirement)
- **PIL/Pillow**: Image I/O operations
- **Matplotlib**: Visualization (Agg backend)
- **scipy**: Morphological operations
- **logging**: Built-in Python logging

## Usage Instructions

### Training
1. Verify dataset path in Config class
2. Adjust hyperparameters if needed
3. Execute training script
4. Monitor logs in `logs/` directory
5. Best model saved automatically to `hrnet_hybrid_best.pth`

### Inference
1. Load checkpoint: `model.load_state_dict(torch.load(path))`
2. Create 5-channel input (RGB + Sobel + Canny)
3. Forward pass: `output = model(input)`
4. Apply softmax and threshold: `pred = (softmax(output)[:, 1] > 0.5)`
5. Post-process: Remove components < 25 pixels

## Key Technical Innovations

1. **Focal Loss**: Addresses class imbalance inherent in coastal imagery
2. **Lovasz-Softmax**: Direct IoU optimization for better boundary accuracy
3. **AdamW Optimizer**: Decoupled weight decay for improved generalization
4. **Cosine Annealing**: Smooth learning rate schedule without manual tuning
5. **Hybrid Edge Channels**: Explicit boundary information for improved shoreline detection
6. **Corrected Sobel**: Bug-fixed edge detection for accurate gradient magnitude
7. **Optimized Post-Processing**: Fine-tuned threshold for detail preservation

## Model Advantages

- **State-of-the-Art Losses**: Focal and Lovasz-Softmax for superior optimization
- **Robust Optimization**: AdamW + Cosine Annealing for stable convergence
- **Explicit Edge Guidance**: 5-channel input with dedicated edge channels
- **High-Resolution Processing**: HRNet maintains spatial precision
- **Multi-Scale Features**: Parallel branches capture context at all scales
- **Fine Detail Preservation**: Lowered post-processing threshold (25 pixels)
- **Comprehensive Evaluation**: Per-k SCR analysis for scale-specific assessment

## Performance Analysis

The model provides comprehensive metrics:
- Per-epoch training loss components breakdown
- Validation metrics (mSCR, IoU, accuracy, precision, recall)
- Per-k SCR scores (k=4, 8, 24, 48) for boundary analysis
- Best model selection based on validation mSCR
- Final test evaluation with all metrics
- Detailed logs for debugging and analysis

## Ablation Study Opportunities

This implementation enables investigation of:
1. **Focal vs. Cross-Entropy**: Impact of class imbalance handling
2. **Lovasz-Softmax vs. Dice**: Boundary optimization comparison
3. **AdamW vs. Adam**: Weight decay effectiveness
4. **Cosine vs. Plateau Scheduler**: Learning rate schedule impact
5. **Edge Channels**: 5-channel vs. 3-channel (RGB only) input
6. **Post-Processing Threshold**: 25 vs. higher thresholds
7. **Loss Weights**: Sensitivity analysis of loss component weights

## Reproducibility Guidelines

For reproducible results:
- Fix random seeds: `torch.manual_seed(42)`, `np.random.seed(42)`
- Enable deterministic algorithms: `torch.use_deterministic_algorithms(True)`
- Document environment: PyTorch version, CUDA version, GPU model
- Version control: Save Config class state for each run
- Log everything: Comprehensive logging enabled by default
- Save checkpoints: Best model and training history

## Citation and Acknowledgments

This implementation builds upon:
- HRNet architecture for high-resolution semantic segmentation
- Focal Loss for addressing class imbalance
- Lovasz-Softmax Loss for IoU optimization
- AdamW optimizer for improved regularization
- CAID dataset for coastal aerial imagery analysis
