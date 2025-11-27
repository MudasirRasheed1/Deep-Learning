# Enhanced HRNet-W48 for Land–Water Segmentation (v2)

## 📌 Overview
This repository implements an **Enhanced HRNet-W48** for high-precision **land vs water** segmentation on the **CAID dataset**, optimized specifically for the **mSCR (Modified Shoreline Conformity Rate)** metric.  
The model uses **RGB + edge channels**, multi-scale attention, a boundary-aware head, and a custom loss pipeline tailored for shoreline extraction.

---

## 🚀 Key Features
### ✔ Hybrid 5-Channel Input  
**RGB + Sobel Magnitude + Canny Edges**

### ✔ HRNet-W48 Backbone  
High-resolution parallel branches with multi-scale fusion.

### ✔ Multi-Stage Attention  
- Channel Attention  
- Spatial Attention  
- CBAM (Channel + Spatial)

### ✔ Boundary-Aware Head  
Auxiliary branch predicting shoreline boundaries directly.

### ✔ Custom Multi-Loss Optimization  
- **Focal Loss** (replaces Cross-Entropy)  
- **Lovasz-Softmax Loss** (replaces Dice Loss)  
- **Differentiable mSCR Loss**  
- **Boundary Loss**

### ✔ Advanced Optimizer & Scheduler  
- **AdamW** optimizer  
- **CosineAnnealingLR** scheduler  
- Mixed precision training (AMP)

---

## 🆕 **Changes from Previous Version (v1 → v2)**
This version includes targeted upgrades while keeping the overall model architecture unchanged.

### 🔧 **1. Bug Fixes**
- **Corrected Sobel magnitude** computation  
  `sobelx*2` → `sobelx**2` (critical fix)

### 📏 **2. Metric Fix**
- SCR computation reverted to the **original formula**  
  → Ensures fair comparison with previous runs

### ⚙ **3. Loss Upgrades**
- **Cross-Entropy → Focal Loss**
- **Dice Loss → Lovasz-Softmax Loss**

### 🚄 **4. Optimizer & Scheduler Upgrades**
- **Adam → AdamW**  
- **ReduceLROnPlateau → CosineAnnealingLR**

### 🧹 **5. Post-Processing Tweak**
- Reduced `MIN_AREA_THRESHOLD` from **100 → 25**  
  → Removes tiny speckles without over-smoothing boundaries

---

## 📊 Training Configuration
- **Batch Size:** 8  
- **Epochs:** 50  
- **Optimizer:** AdamW (lr=1e-4, wd=1e-2)  
- **Scheduler:** CosineAnnealingLR  
- **Input:** 500×500 (5-channels)  
- **Mixed Precision:** Yes