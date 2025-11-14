# Enhanced HRNet-W48 for Land-Water Segmentation

## 📋 Overview

This project implements an **Enhanced HRNet-W48 (High-Resolution Network)** with attention mechanisms and boundary-aware supervision for precise land vs water semantic segmentation on the CAID (Coastal Aerial Image Dataset). The model is specifically optimized for the **mSCR (modified Shoreline Conformity Rate)** metric, which measures how accurately the model predicts shoreline boundaries at multiple neighborhood scales.

## 🎯 Problem Statement

Traditional semantic segmentation models optimize for pixel-wise accuracy or IoU, but **shoreline detection requires boundary-level precision**. The mSCR metric evaluates predictions at different tolerance levels (k=4, 8, 24, 48 pixels), making it crucial to:

1. **Accurately detect shoreline pixels** (land-water boundaries)
2. **Maintain spatial precision** at multiple scales
3. **Leverage edge information** from the imagery

## 🏗️ Architecture

### Core Components

#### 1. **Hybrid 5-Channel Input**
```
Input: RGB + Sobel + Canny Edge Maps
├── RGB Channels (3) - Color information
├── Sobel Magnitude (1) - Gradient-based edges
└── Canny Edges (1) - Binary edge map
```

**Why?** Edge channels provide explicit boundary cues that complement RGB features for shoreline detection.

#### 2. **HRNet-W48 Backbone**

HRNet maintains high-resolution representations throughout the network via parallel multi-resolution branches:

```
Stage 1: [64 channels]
Stage 2: [48, 96 channels] (2 parallel branches)
Stage 3: [48, 96, 192 channels] (3 parallel branches)
Stage 4: [48, 96, 192, 384 channels] (4 parallel branches)
```

**Key Features:**
- Maintains high-resolution features (no aggressive downsampling)
- Multi-scale feature extraction through parallel branches
- Repeated multi-scale fusion between branches

#### 3. **Attention Mechanisms**

##### a) **Channel Attention (SE-style)**
```python
Input → Global Avg Pool + Global Max Pool
      → FC Layers (squeeze to channels/16, then expand)
      → Sigmoid → Channel weights
```

**Purpose:** Learn to emphasize edge channels (Sobel/Canny) when detecting boundaries and RGB channels for texture/context.

##### b) **Spatial Attention**
```python
Input → Channel-wise Avg + Channel-wise Max
      → Conv (2→1 channels)
      → Sigmoid → Spatial weights
```

**Purpose:** Focus on boundary regions (shorelines) which are sparse (~1-2% of pixels) but critical for mSCR.

##### c) **CBAM (Convolutional Block Attention Module)**
Combines both channel and spatial attention sequentially:
```
Input → Channel Attention → Spatial Attention → Output
```

**Applied at:**
- Input stem (weight RGB vs edges)
- All Stage 4 outputs (refine multi-scale features)
- Multi-scale fusion module

#### 4. **Boundary Refinement Module**

```python
Input Features
├── Standard Conv (3×3, padding=1)
└── Dilated Conv (3×3, dilation=2)  # Larger receptive field
    └── Concatenate with input → 1×1 fusion
```

**Purpose:** 
- Dilated convolutions capture larger context around boundaries
- Edge-aware processing for sharper shoreline predictions

#### 5. **Enhanced Multi-Scale Fusion**

```python
For each HRNet branch output:
    1. Upsample to highest resolution
    2. Process through Conv + BN + ReLU + CBAM
    3. Concatenate all processed features
    4. Apply fusion convolution with CBAM
    5. Apply boundary refinement module
```

**Output:** 256-channel fused feature map with boundary-focused representations

#### 6. **Dual-Head Architecture**

##### Main Segmentation Head
```
Fused Features (256) 
→ Conv (256→128) + BN + ReLU + Dropout(0.1)
→ Conv (128→2) [Land vs Water logits]
```

##### Auxiliary Boundary Head
```
Fused Features (256)
→ Conv (256→64) + BN + ReLU
→ Conv (64→1) [Boundary logits]
```

**Purpose:** Explicit boundary supervision forces the network to learn boundary-aware features.

## 📊 Loss Functions

### 1. **Cross-Entropy Loss (Weight: 1.0)**
Standard pixel-wise classification loss.

### 2. **Dice Loss (Weight: 1.0)**
```python
Dice = 2 × |Pred ∩ Target| / (|Pred| + |Target|)
Loss = 1 - Dice
```
Handles class imbalance (land vs water may be unbalanced).

### 3. **Differentiable mSCR Loss (Weight: 0.5)**
```python
For each k ∈ {4, 8, 24, 48}:
    1. Extract shorelines using Laplacian edge detection
    2. Expand shorelines by k-neighborhood
    3. Compute bidirectional SCR
    
mSCR = mean(SCR_k4, SCR_k8, SCR_k24, SCR_k48)
Loss = 1 - mSCR
```

**Differentiability:** Uses soft operations (convolutions, max pooling) instead of hard morphological operations.

### 4. **Boundary-Aware Loss (Weight: 0.3)**
```python
Ground Truth Boundaries = Extract shoreline from masks
Predicted Boundaries = Boundary head output
Loss = BCEWithLogitsLoss(predicted, ground_truth)
```

**Purpose:** Direct supervision for boundary pixels.

### Total Loss
```
Total = 1.0 × CE + 1.0 × Dice + 0.5 × mSCR + 0.3 × Boundary
```

## 📈 Metrics

### 1. **Pixel Accuracy**
Standard classification accuracy across all pixels.

### 2. **IoU (Intersection over Union)**
```
IoU_class = |Pred ∩ Target| / |Pred ∪ Target|
Mean IoU = (IoU_land + IoU_water) / 2
```

### 3. **mSCR (Modified Shoreline Conformity Rate)**

**Evaluation Process:**
1. Extract shoreline pixels from prediction and ground truth
2. For each k ∈ {4, 8, 24, 48}:
   - Expand shorelines by k-neighborhood
   - Compute bidirectional SCR
3. Average across all k values

**Interpretation:**
- **k=4**: Strict evaluation (±4 pixels tolerance)
- **k=8**: Moderate tolerance
- **k=24**: Lenient tolerance
- **k=48**: Very lenient tolerance

Higher k → higher scores (more tolerance for boundary errors)

### 4. **Per-k SCR Scores**
Individual scores at each k value to diagnose boundary precision at different scales.

## 🚀 Training Configuration

```python
Batch Size: 8
Epochs: 50
Optimizer: Adam (lr=1e-4, weight_decay=1e-4)
Scheduler: ReduceLROnPlateau (factor=0.5, patience=5)
Mixed Precision: Enabled (AMP)
```

### Data Augmentation
- Random horizontal flip (p=0.5)
- Random vertical flip (p=0.5)
- Random rotation: 90°, 180°, or 270° (p=0.3)

### Input Normalization
- **RGB channels:** ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- **Edge channels:** Normalize from [0, 1] to [-1, 1]

## 📁 File Structure

```
Model-4/
├── code.py                  # Main training script
├── README.md                # This file
├── logs/                    # Training logs
│   └── training_YYYYMMDD_HHMMSS.log
├── hrnet_hybrid_best.pth    # Best model checkpoint
├── training_curves.png      # Visualization of metrics
└── training_history.csv     # Detailed epoch-wise metrics
```

## 🔧 Usage

### Training

```python
# Configure paths in Config class
DATA_ROOT = '/kaggle/input/caid-dataset/CAID'
OUTPUT_DIR = '/kaggle/working'

# Run training
python code.py
```

### Model Inference

```python
import torch
from code import HRNetW48, create_hybrid_input
from PIL import Image
import numpy as np

# Load model
model = HRNetW48(num_classes=2, input_channels=5)
model.load_state_dict(torch.load('hrnet_hybrid_best.pth'))
model.eval()
model.to('cuda')

# Load and prepare image
image = Image.open('test_image.png').convert('RGB')
hybrid_input = create_hybrid_input(image)  # Create 5-channel input

# Normalize
rgb = hybrid_input[:, :, :3] / 255.0
edges = hybrid_input[:, :, 3:]
rgb_normalized = (rgb - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
edges_normalized = edges * 2.0 - 1.0
final_input = np.concatenate([rgb_normalized, edges_normalized], axis=2)

# Convert to tensor
input_tensor = torch.from_numpy(final_input).permute(2, 0, 1).unsqueeze(0).float().to('cuda')

# Inference
with torch.no_grad():
    seg_out, boundary_out = model(input_tensor)
    prediction = torch.argmax(seg_out, dim=1).squeeze().cpu().numpy()
    
# prediction: 0=Land, 1=Water
```

## 📊 Expected Performance

| Metric | Expected Range | Notes |
|--------|---------------|-------|
| Pixel Accuracy | 95-98% | High due to large homogeneous regions |
| Mean IoU | 85-92% | Good overall segmentation quality |
| **mSCR** | **60-75%** | **Primary optimization target** |
| SCR k=4 | 45-60% | Strict boundary evaluation |
| SCR k=8 | 55-70% | Moderate boundary evaluation |
| SCR k=24 | 70-85% | Lenient boundary evaluation |
| SCR k=48 | 80-90% | Very lenient boundary evaluation |

## 🔍 Key Innovations

1. **Hybrid Input:** Explicit edge channels (Sobel + Canny) complement RGB
2. **Multi-level Attention:** Channel, spatial, and combined (CBAM) at multiple stages
3. **Boundary-Aware Training:** Dual-head architecture with explicit boundary supervision
4. **Enhanced Fusion:** Attention-weighted multi-scale feature aggregation
5. **Boundary Refinement:** Dilated convolutions for edge-aware processing
6. **Custom Loss:** Differentiable mSCR loss + boundary loss for direct optimization

## 🐛 Known Limitations

1. **Memory Intensive:** HRNet-W48 with attention requires significant GPU memory (batch size 8 on ~16GB GPU)
2. **Computational Cost:** Multiple attention modules increase inference time (~2-3x slower than baseline HRNet)
3. **Edge Detection Sensitivity:** Sobel/Canny parameters may need tuning for different datasets
4. **Training Time:** ~8-10 hours for 50 epochs on single V100 GPU

## 🔮 Future Improvements

1. **Conditional Random Fields (CRF):** Post-processing for boundary smoothing
2. **Test-Time Augmentation:** Multi-scale testing for improved robustness
3. **Active Boundary Sampling:** Oversample boundary regions during training
4. **Curriculum Learning:** Start with k=48, gradually focus on k=4
5. **Edge-Conditioned Convolutions:** Learn edge-specific filters
6. **Knowledge Distillation:** Compress model for faster inference

## 📚 References

1. **HRNet:** Wang et al., "Deep High-Resolution Representation Learning for Visual Recognition" (CVPR 2019)
2. **CBAM:** Woo et al., "CBAM: Convolutional Block Attention Module" (ECCV 2018)
3. **Boundary-Aware Networks:** Takikawa et al., "Gated-SCNN: Gated Shape CNNs for Semantic Segmentation" (ICCV 2019)
4. **Edge Detection:** Canny, J., "A Computational Approach to Edge Detection" (IEEE TPAMI 1986)

## 🛠️ Troubleshooting

### Out of Memory Error
- Reduce batch size to 4 or 2
- Use gradient checkpointing
- Disable mixed precision training

### Low mSCR Scores
- Increase boundary loss weight (0.3 → 0.5 or 1.0)
- Tune Sobel/Canny edge detection parameters
- Add CRF post-processing
- Check if edge channels are properly normalized

### Training Instability
- Reduce learning rate (1e-4 → 5e-5)
- Increase weight decay for regularization
- Use gradient clipping

## 📧 Contact

For questions or issues, please contact [your email/team].

---

**Model Parameters:** ~21.7M  
**Input Size:** 500×500×5  
**Output:** 500×500 (2 classes: Land=0, Water=1)  
**Framework:** PyTorch 2.0+  
**CUDA Required:** Yes (GPU training essential)

**Last Updated:** January 2025
