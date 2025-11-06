# Model-3 — HRNet-W48 with Hybrid Edge Channels for CAID

This folder contains an implementation and training script that adapts an HRNet-W48-style network for binary semantic segmentation (land vs water) on the CAID dataset. The model supports a hybrid input: RGB channels augmented with two edge channels (Sobel magnitude and Canny edges).

Files
- `code.py` — End-to-end training script implementing:
  - `CAIDHybridDataset` which creates 5-channel input (`R,G,B,Sobel,Canny`) when `cfg.USE_EDGE_CHANNELS=True`.
  - `HRNetW48` lightweight HRNet-style network with multi-resolution branches and a final classifier.
  - Edge-detection utilities: `compute_sobel_magnitude` and `compute_canny_edges` (OpenCV-based).
  - Modified mSCR evaluation (`compute_scr_by_k_for_batch`) that computes per-k SCR means and an overall mSCR (per-k averaging).
  - Differentiable mSCR loss (`mSCRLoss`) and combined loss (CE + Dice + mSCR).
  - Training/validation loops, logging, and a `MetricsPlotter` that tracks differentiable vs evaluation mSCR and per-k SCR curves.

Highlights
- Hybrid input: enriches RGB with explicit edge channels — can improve boundary conformity for shoreline segmentation.
- Two mSCR variants:
  - Differentiable: used in the loss to guide training (PyTorch operations, pooling-based expansions).
  - Evaluation: NumPy + morphological operations that compute SCR per k and then average across k-values (per-k averaging) — reported and used for model selection.

Configuration
- `Config` class at top of `code.py` controls:
  - Data paths (`DATA_ROOT`, `IMAGES_DIR`, `MASKS_DIR`, `TRAIN_TXT`, `VAL_TXT`)
  - Model/hyperparams (`BATCH_SIZE`, `NUM_EPOCHS`, `LEARNING_RATE`, `BACKBONE`)
  - Hybrid input toggle: `USE_EDGE_CHANNELS` and `INPUT_CHANNELS` (set automatically)
  - Edge detector params: `SOBEL_KSIZE`, `CANNY_THRESHOLD1`, `CANNY_THRESHOLD2`
  - mSCR neighborhood sizes: `K_NEIGHBORS = [4,8,24,48]`

Dependencies
- Python 3.8+
- PyTorch + torchvision
- numpy, pandas, pillow, matplotlib, tqdm
- scipy, opencv-python (cv2), IPython (optional for notebook visualization)

Quick start
1. Install dependencies (example):

```powershell
pip install torch torchvision numpy pandas pillow matplotlib tqdm scipy opencv-python
```

2. If not on Kaggle, update `Config` paths to point to your dataset.

3. Run training:

```powershell
python code.py
```

Notes & tips
- To disable edge channels and use standard RGB input, set `cfg.USE_EDGE_CHANNELS = False` (and restart). The model will expect 3 channels.
- The hybrid pipeline normalizes RGB channels with ImageNet stats and maps edge channels to [-1, 1]. Make sure any external preprocessing matches this.
- If mSCR evaluation (`mscr_eval`) is used for model selection, ensure the metrics dictionary keys match exactly (the code uses `mscr_eval` lower-case). If you integrate different scripts, be careful with key names to avoid KeyError.
- HRNet-W48 here is a simplified, self-contained variant (not a direct copy of official repo). Parameter counts will be smaller; adjust `BATCH_SIZE` accordingly.

Suggested experiments
- Ablation: compare with and without edge channels to see boundary improvements.
- Tune Canny thresholds and Sobel kernel size to better match the dataset characteristics.
- Replace max-pool dilation in differentiable mSCR with custom structuring elements (to match evaluation implementation) for better alignment between training signal and evaluation.

Outputs
- Best model: `cfg.BEST_MODEL_PATH` (saved when validation mSCR improves).
- Logs: saved under `cfg.LOG_DIR` with a timestamped log file.
- Training curves and history CSV saved to `cfg.OUTPUT_DIR`.

Troubleshooting
- Missing files / paths: verify `TRAIN_TXT`/`VAL_TXT` contain correct image IDs and the images/masks are present.
- OOM: HRNet can be memory heavy; reduce `BATCH_SIZE` or disable edge channels temporarily.
- If the differentiable mSCR (`mscr_diff`) and evaluation mSCR (`mscr_eval`) diverge, consider the suggestions above (unify shoreline extraction and expansion methods).
