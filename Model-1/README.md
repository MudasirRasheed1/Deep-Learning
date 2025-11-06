# Model-1 — EncNet (Context Encoding Network) for CAID

This folder contains an implementation and training script for EncNet applied to the CAID (Coastal Aerial Imagery Dataset) for binary semantic segmentation (land vs water).

## Files
- `code.py` — Full training script implementing:
  - Dataset loader `CAIDDataset` (synchronized augmentations and ImageNet normalization)
  - `EncNet` model (ResNet backbone + Context Encoding Module + segmentation head)
  - Differentiable mSCR loss (`mSCRLoss`) and traditional losses (CrossEntropy + Dice)
  - Corrected non-differentiable evaluation mSCR metric (per-image average across k-values -> batch mean)
  - Training/validation loop, metric logging, and a `MetricsPlotter` for visualizing loss components and SCR per-k.

## Overview
- Purpose: train an EncNet segmentation model on CAID to optimize both pixel-wise accuracy/IoU and shoreline conformity using the mSCR metric.
- The script uses a combined loss: CE + Dice + (1 - mSCR) where mSCR is a differentiable approximation implemented in PyTorch.
- Evaluation uses a corrected, non-differentiable mSCR computed on binarized predictions using morphological operations.

## Key components
- `Config` class: central place for dataset paths, hyperparameters (batch size, lr, epochs), augmentation probabilities, and mSCR neighborhood sizes `K_NEIGHBORS = [4,8,24,48]`.
- `ContextEncoding` / `EncModule`: implements codebook-based context encoding and a channel attention head (like SE block).
- `mSCRLoss`: computes SCR on soft/binarized predictions for k neighborhoods, averages across k and batch, and returns loss = 1 - mSCR.
- `compute_mscr_batch`: corrected evaluation mSCR implementation that returns per-k SCRs and batch mSCR.

## Dependencies
- Python 3.8+ (tested in Kaggle-like environment)
- PyTorch (>=1.10 recommended)
- torchvision
- numpy, pandas, pillow, matplotlib, tqdm
- scipy (for `binary_dilation`), IPython (for notebook display)

## Typical usage
1. Install dependencies (example):

```powershell
# Windows PowerShell example
pip install torch torchvision numpy pandas pillow matplotlib tqdm scipy opencv-python
```

2. Update dataset paths in the `Config` class (if not using Kaggle):
   - `DATA_ROOT`, `IMAGES_DIR`, `MASKS_DIR`, `TRAIN_TXT`, `VAL_TXT`.

3. Run the training script:

```powershell
python code.py
```

## What the script produces
- Model checkpoints: saved when validation mSCR improves (path = `cfg.BEST_MODEL_PATH`).
- Training curves image: `training_curves.png` in `cfg.OUTPUT_DIR`.
- Training history CSV: `training_history.csv` (losses, metrics per epoch).

## Notes & troubleshooting
- If you get file path errors, ensure `TRAIN_TXT`/`VAL_TXT` contain image IDs (without `.png`) and the images/masks exist at `IMAGES_DIR`/`MASKS_DIR`.
- mSCR: two different implementations exist: differentiable (`mSCRLoss`) used for training and corrected evaluation (`compute_mscr_batch`) used for monitoring. Differences may be due to thresholding, structuring elements, and denominators — see code comments.
- Mixed precision: enabled by default (`USE_MIXED_PRECISION = True`), but you can set it to `False` for CPU-only runs.
- GPU/Memory: ResNet101 + EncNet is heavy; reduce `BATCH_SIZE` or use a smaller backbone in `Config` if you hit OOM.

## Suggestions for improvements
- Unify shoreline extraction between the differentiable and evaluation implementations for more consistent signals (see repo comments).
- Add a small unit-test script to run the model forward pass on a random tensor to confirm installation and shapes.

## Contact / Attribution
- Script written to experiment on CAID with EncNet and mSCR metrics. Use freely for research; cite or note modifications where appropriate.
