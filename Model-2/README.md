# Model-2 — EncNet (Duplicate/Variant) for CAID

This folder contains an EncNet training script very similar to Model-1. It implements the same architecture and metrics tailored for the CAID dataset.

Files
- `code.py` — Training script with the following main features:
  - `CAIDDataset` loader with augmentations and normalization
  - `EncNet` (ResNet backbone + Context Encoding Module + segmentation head)
  - Differentiable `mSCRLoss` and traditional Dice/Cross-Entropy losses
  - Corrected evaluation mSCR (`compute_mscr_batch`) and per-k SCR reporting
  - Training / validation loops and live plotting via `MetricsPlotter`.

Why there are two EncNet folders
- `Model-2` appears to be a near-duplicate or slight variant of `Model-1`. Keep both if you are experimenting with slightly different defaults or to preserve separate experiment histories. If they are identical, you can consolidate to a single folder.

Quick start
1. Install dependencies (same as Model-1):

```powershell
pip install torch torchvision numpy pandas pillow matplotlib tqdm scipy opencv-python
```

2. Edit paths in `Config` (if not running on Kaggle) and run:

```powershell
python code.py
```

Important settings
- `Config` contains the main knobs: `BATCH_SIZE`, `NUM_EPOCHS`, `LEARNING_RATE`, `BACKBONE` (e.g., `resnet50`/`resnet101`), `K_NEIGHBORS` for mSCR, and `USE_MIXED_PRECISION`.

Notes
- Same troubleshooting and improvement suggestions as Model-1 apply (see Model-1 README).
