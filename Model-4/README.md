# Model 4: Seg-GAN - Adversarial Segmentation Network

## Overview

This implementation presents an adversarial training framework for semantic segmentation of the CAID (Coastal Aerial Imagery Dataset). The model employs a Generative Adversarial Network (GAN) approach where a U-Net++ generator produces segmentation maps and a PatchGAN discriminator evaluates their realism, optimizing for both segmentation accuracy and boundary realism.

## Architecture

### Generator: U-Net++ with EfficientNet-B5 Encoder
- **Architecture**: U-Net++ (nested U-Net with dense skip connections)
- **Encoder**: EfficientNet-B5 pretrained on ImageNet
- **Skip Connections**: Nested dense connections for multi-scale feature propagation
- **Decoder**: Progressive upsampling with feature concatenation
- **Output**: 2-channel segmentation logits (land and water)

### Discriminator: PatchGAN
- **Architecture**: Fully convolutional discriminator
- **Input**: Concatenation of RGB image (3 channels) and segmentation logits (2 channels)
- **Receptive Field**: 70×70 patches (PatchGAN)
- **Structure**: 5 convolutional blocks with stride-2 downsampling
- **Output**: Patch-wise real/fake predictions
- **Activation**: LeakyReLU (0.2) with batch normalization

### Discriminator Architecture Details
```
Block 1: Conv(5→64, 4×4, s=2) + LeakyReLU
Block 2: Conv(64→128, 4×4, s=2) + BatchNorm + LeakyReLU
Block 3: Conv(128→256, 4×4, s=2) + BatchNorm + LeakyReLU
Block 4: Conv(256→512, 4×4, s=2) + BatchNorm + LeakyReLU
Block 5: ZeroPad + Conv(512→1, 4×4)
```

## Training Configuration

### Hyperparameters
- **Batch Size**: 8
- **Total Epochs**: 12
- **Generator Learning Rate**: 1e-4
- **Discriminator Learning Rate**: 1e-4
- **Beta1**: 0.5
- **Beta2**: 0.999
- **Input Resolution**: 512 × 512 pixels

### Generator Loss Function
Multi-objective loss with four components:
- **Cross-Entropy Loss** (weight: 1.0): Pixel-wise classification
- **Dice Loss** (weight: 1.0): Region overlap optimization
- **Differentiable mSCR Loss** (weight: 0.5): Boundary conformity
- **Adversarial Loss** (weight: 0.1): Fool discriminator objective

### Discriminator Loss Function
- **LSGAN Loss**: Least Squares GAN for stable training
- **Real Samples**: MSE loss with target 1.0
- **Fake Samples**: MSE loss with target 0.0

### Data Augmentation (Albumentations)
- Resize to 512×512 (INTER_NEAREST for masks)
- Horizontal flip (probability: 0.5)
- Vertical flip (probability: 0.5)
- Random rotation (±35°, probability: 0.3)
- Random brightness and contrast (probability: 0.3)
- Gaussian noise (probability: 0.2)
- CoarseDropout (8 holes, 32×32, probability: 0.3)
- ImageNet normalization

## Training Strategy

### Adversarial Training Loop
1. **Discriminator Update**:
   - Forward pass with real image and ground truth mask
   - Forward pass with real image and generated mask
   - Compute LSGAN loss for real and fake samples
   - Backpropagate and update discriminator

2. **Generator Update**:
   - Generate segmentation prediction
   - Compute segmentation losses (CE + Dice + mSCR)
   - Compute adversarial loss from discriminator feedback
   - Combine losses with specified weights
   - Backpropagate and update generator

### Mixed Precision Training
- Automatic Mixed Precision (AMP) enabled
- Separate GradScalers for generator and discriminator
- Improved training speed and memory efficiency

## Evaluation Metrics

### Mean Shoreline Conformity Rate (mSCR)
- **Neighborhood Sizes**: k ∈ {4, 8, 24, 48}
- **Bidirectional Evaluation**: Both prediction-to-GT and GT-to-prediction
- **Per-k SCR**: Individual scores for each neighborhood size
- **Final mSCR**: Average across all k values

### Additional Metrics
- **Intersection over Union (IoU)**: Per-class and mean IoU
- **Pixel Accuracy**: Overall classification accuracy
- **Precision and Recall**: Class-specific performance metrics

## Dataset Specifications

### CAID Dataset
- **Data Root**: `/kaggle/input/caid-dataset/CAID`
- **Images**: `JPEGImages/` (PNG format)
- **Masks**: `SegmentationClass/` (PNG format, binary)
- **Splits**: train.txt, val.txt, test.txt
- **Classes**: 0 (Land), 1 (Water)

### Data Loading
- Custom Dataset class with Albumentations transforms
- Synchronized augmentation for images and masks
- Multi-threaded loading (2 workers)
- Pin memory for faster GPU transfer

## Implementation Details

### Differentiable mSCR Loss
- **Shoreline Extraction**: Laplacian kernel-based edge detection
- **Neighborhood Expansion**: Max pooling with varying kernel sizes
- **Soft Predictions**: Uses softmax probabilities for gradient flow
- **Bidirectional SCR**: Averages prediction-to-GT and GT-to-prediction

### LSGAN Adversarial Loss
- Uses Mean Squared Error instead of cross-entropy
- More stable training compared to standard GAN
- Provides stronger gradients for generator

### Model Checkpointing
- Saves generator state only (discriminator not needed for inference)
- Best model selected based on validation mSCR
- Checkpoint path: `seg_gan_generator_best.pth`

## Computational Requirements

- **Platform**: CUDA-enabled GPU
- **Memory**: High VRAM requirement for U-Net++ with EfficientNet-B5 and discriminator
- **Dependencies**: PyTorch, segmentation-models-pytorch, albumentations, OpenCV, NumPy

### Library Installation
```bash
pip install -q segmentation-models-pytorch albumentations
pip install -q "numpy<2.0" --force-reinstall
```

## Usage

### Training Process
1. Initialize generator (U-Net++) and discriminator (PatchGAN)
2. Alternate between discriminator and generator updates
3. Validate after each epoch using generator only
4. Save best generator model based on validation mSCR
5. Monitor loss components and metrics

### Inference
1. Load trained generator checkpoint
2. Process image through generator
3. Output segmentation map (logits or softmax probabilities)
4. Discriminator not required during inference

## Key Innovations

1. **Adversarial Training**: Discriminator feedback improves boundary realism
2. **U-Net++ Architecture**: Dense skip connections for multi-scale features
3. **EfficientNet-B5 Encoder**: Powerful pretrained feature extractor
4. **PatchGAN Discriminator**: Local patch-based realism evaluation
5. **Multi-Objective Generator Loss**: Balances accuracy, overlap, boundary, and realism
6. **LSGAN Framework**: Stable adversarial training

## Model Advantages

- **Realistic Boundaries**: Adversarial loss encourages sharp, realistic segmentation edges
- **Strong Pretrained Encoder**: EfficientNet-B5 provides robust feature extraction
- **Dense Feature Propagation**: U-Net++ architecture captures multi-scale information
- **Comprehensive Augmentation**: Albumentations library provides diverse transformations
- **Stable Training**: LSGAN reduces mode collapse and training instability

## Performance Analysis

The model reports:
- Per-epoch generator and discriminator losses
- Validation metrics (mSCR, IoU, accuracy)
- Per-k SCR breakdown for boundary analysis
- Best validation checkpoint
- Final test set evaluation

## Hyperparameter Considerations

- **Adversarial Loss Weight (0.1)**: Balanced to avoid overwhelming segmentation objectives
- **Reduced Epochs (12)**: Faster experimentation with adversarial training
- **Larger Input Size (512×512)**: Better resolution for boundary details
- **LSGAN Loss**: More stable than standard GAN cross-entropy loss

## Ablation Studies

The adversarial approach can be evaluated by:
- Comparing with non-adversarial U-Net++ baseline
- Analyzing discriminator loss convergence
- Evaluating boundary sharpness improvements
- Measuring mSCR gains from adversarial training
