# Seg-GAN: Adversarial Segmentation for the CAID Dataset
This project implements an advanced adversarial segmentation network, "Seg-GAN," designed for high-fidelity semantic segmentation of land vs. water on the CAID dataset.

This model employs a Conditional Generative Adversarial Network (cGAN) framework. Instead of relying *only* on traditional pixel-wise segmentation losses (which can result in correct but blurry or unrealistic boundaries), this approach adds a "realism critic." This forces the segmentation model to produce masks that are not only pixel-perfect but also structurally plausible and sharp, a critical requirement for complex boundaries like shorelines.

* **Generator (G):** A `U-Net++` model with an `EfficientNet-B5` encoder. Its job is to perform the segmentation, generating a mask from an input image.
* **Discriminator (D):** A `PatchGAN` discriminator. Its job is to act as a critic, distinguishing between *real* (image, ground-truth mask) pairs and *fake* (image, generated mask) pairs.

---

## Project Goal & Core Concept

The primary challenge in shoreline segmentation is not just classifying pixels as "land" or "water," but precisely capturing the **complex, fractal-like boundary** between them.

**Why a standard segmentation model isn't enough:**
Traditional models optimized with pixel-wise losses (like Cross-Entropy or Dice) are incentivized to maximize overlap. This often leads to:
* **Overly-smoothed boundaries:** The model "hedges its bets" at the boundary, creating blurry, averaged predictions.
* **Unrealistic artifacts:** The model may fail to connect small islands or create "blob-like" shapes that don't look like real shorelines.

**Why Seg-GAN is the chosen architecture:**
This project reframes the problem as a cGAN (Conditional GAN):
> The Generator's goal is to create a segmentation mask (conditioned on the input image) that is **indistinguishable** from a real, human-annotated mask.

This adversarial setup creates a powerful two-player game:
1.  The **Generator** (U-Net++) is forced to produce sharp, clean, and structurally realistic shorelines to "fool" the critic.
2.  The **Discriminator** (PatchGAN) becomes a highly specialized critic, learning the subtle textures and patterns that make a shoreline "real" vs. "fake."

This dual objective—optimizing for **pixel-wise accuracy** (via segmentation loss) and **perceptual realism** (via adversarial loss)—is the core strategy of this model.

## Architectural Deep Dive

The architecture consists of two distinct neural networks.



### 1. The Generator (G): U-Net++ with EfficientNet-B5

* **Model:** `smp.UnetPlusPlus(encoder_name='efficientnet-b5', ...)`
* **Purpose:** This is the primary segmentation model. It takes a 3-channel RGB image and outputs 2-channel logits (one for "Land", one for "Water").

**Why this architecture was chosen:**

* **Why `U-Net++`?**
    Standard U-Nets use skip connections to combine shallow, high-resolution features (good for *location*) with deep, low-resolution features (good for *context*). `U-Net++` improves this by creating **nested and dense skip pathways**. This allows the network to aggregate features at multiple scales *before* fusing them at the decoder, which drastically reduces the semantic gap between the encoder and decoder. For a problem like shorelines, this is ideal for capturing both large landmasses and tiny, intricate water inlets.

* **Why `EfficientNet-B5` as the Encoder?**
    The encoder acts as the model's "eyes." Instead of training one from scratch, we use `EfficientNet-B5` pre-trained on ImageNet.
    * **Power:** `EfficientNet` is a state-of-the-art classifier that provides an exceptionally rich and high-quality set of features for the decoder to use.
    * **Variant:** 'B5' is a large, high-capacity variant. Given the complexity of the task, we opt for a powerful (though slower) backbone to maximize feature extraction quality.

### 2. The Discriminator (D): PatchGAN

* **Model:** Custom `Discriminator` class.
* **Purpose:** This is the "realism critic." It takes a 5-channel input and outputs a 2D "patch" of scores, indicating how "real" it thinks each part of the input is.

**Why this architecture was chosen:**

* **Why 5 Channels? (The "Conditional" Part)**
    The input to the Discriminator is `torch.cat((img, mask_logits), dim=1)`. This is `3 (RGB Image) + 2 (Mask Logits) = 5 channels`.
    * **Reasoning:** This is the *most critical* design choice. The Discriminator is not just asking, "Does this mask look realistic?" It's asking, "**Is this a realistic mask *for this specific image*?**" By seeing both the image and the mask, it learns the conditional probability. It learns that a "water" pixel is plausible *here* (on the sea) but not *there* (in the middle of a desert).

* **Why a `PatchGAN`?**
    A standard discriminator might output a single score (0 or 1) for the *entire* image. This is inefficient.
    * **Reasoning:** A `PatchGAN` (implemented as a fully convolutional network) outputs a 2D grid of scores (e.g., 30x30). Each "pixel" in this output grid corresponds to a larger "patch" (e.g., 70x70) of the input image.
    * **Benefit:** This design forces the Generator to make *every local patch* realistic. It can't get away with one "bad" area. This is a much stronger, more stable, and more localized loss signal that is perfect for penalizing unrealistic boundary segments.

## Data & Augmentation Pipeline

* **Loader:** `CAIDDataset` class.
* **Library:** `Albumentations` is used for high-performance augmentations.

**Why this pipeline was chosen:**

* **Reasoning:** GANs are notoriously data-hungry and can easily "mode collapse" (i.e., the Generator learns one trick to fool the D) or "overfit" (i.e., the Discriminator just memorizes the training set).
* **The Solution:** We use a *very* strong augmentation pipeline to create a robust, varied training set.
    * `Resize(512, 512)`: Standardizes input size.
    * `HorizontalFlip` / `VerticalFlip` / `Rotate`: Provides spatial invariance.
    * `RandomBrightnessContrast` / `GaussNoise`: Provides robustness to different lighting and sensor conditions.
    * `CoarseDropout`: This is a powerful augmentation that randomly cuts out large squares from the image. This forces the model to learn from *context* and prevents it from over-relying on any single feature, making it more robust.

---

## The Loss Function: A Multi-Objective Strategy

The entire system is balanced by two competing loss functions, one for each "player." The chosen adversarial loss is `nn.MSELoss`, which turns the model into a **Least-Squares GAN (LSGAN)**.

**Why an LSGAN (MSELoss) instead of a standard GAN (BCELoss)?**
* **Reasoning:** Standard sigmoid cross-entropy loss can "saturate," meaning the gradients become very small when the Discriminator is too confident. This can cause the Generator's gradients to vanish, effectively stopping it from learning.
* **Benefit:** The MSE (L2) loss is non-saturating. It penalizes predictions that are far from the target (0 or 1) much more heavily, providing smoother, more stable gradients throughout training. This prevents vanishing gradients and leads to a more stable adversarial game.

### Generator Loss (Total)
The Generator has two jobs: be **accurate** and be **realistic**. Its loss is a weighted sum of these two goals.

`loss_G = loss_G_seg + (LOSS_WEIGHT_ADVERSARIAL * loss_G_adv)`

**1. `loss_G_adv` (The "Realism" Loss)**
* **Formula:** `loss_G_adv = MSELoss(D(X, G(X)), 1.0)`
* **Why:** The Generator's goal is to fool the Discriminator. It trains itself by calculating the loss it would get if the Discriminator *believed* its fake masks were real (i.e., output `1.0`). The gradient from this loss teaches the Generator *how* to change its output to better fool the critic.

**2. `loss_G_seg` (The "Accuracy" Loss)**
This is *also* a weighted combination, designed to be robust.

`loss_G_seg = (weight_ce * CE) + (weight_dice * Dice) + (weight_mscr * mSCR)`

* **Why `CrossEntropyLoss (CE)`?**
    * This is the "workhorse" segmentation loss. It's pixel-wise, stable, and easy to optimize.
    * *Weakness:* It struggles with class imbalance (e.g., 90% land, 10% water) and doesn't care about the *shape* of the segmented region.

* **Why `DiceLoss`?**
    * This is a region-based loss (mathematically similar to the F1-score). It measures the *overlap* between the predicted and ground-truth regions.
    * *Benefit:* It's excellent for class imbalance (a 90/10 split is handled naturally) and directly optimizes for the IoU metric. It forces the *shape* of the prediction to be correct.

* **Why `mSCRLoss`?**
    * This is the "secret sauce" for this problem. It's a custom, differentiable loss that *directly* measures boundary quality.
    * *Benefit:* While Dice ensures the "blob" is in the right place, `mSCRLoss` ensures the *edge* of the blob is correct. It uses pooling operations (`expand_neighborhood_torch`) to approximate the mSCR evaluation metric, providing a direct gradient signal for shoreline compliance.

**The Synergy:** By combining **CE** (pixel-wise stability) + **Dice** (region/shape accuracy) + **mSCR** (boundary accuracy), we create an extremely robust `loss_G_seg` that covers all bases *before* the adversarial loss even kicks in.

### Discriminator Loss (Total)
The Discriminator has one job: get better at "spotting fakes."

`loss_D = (loss_D_real + loss_D_fake) * 0.5`

* **`loss_D_real = MSELoss(D(X, Y_real), 1.0)`**
    * **Why:** This trains the Discriminator to output `1.0` (real) when it sees a *real* image paired with its *real* ground-truth mask.

* **`loss_D_fake = MSELoss(D(X, G(X).detach()), 0.0)`**
    * **Why:** This trains the Discriminator to output `0.0` (fake) when it sees a *real* image paired with the Generator's *fake* mask.
    * **Note on `.detach()`:** This is crucial. It stops gradients from flowing back into the Generator during the Discriminator's training step.

---

## Training & Adversarial Process

The training is an alternating optimization loop, executed in `train_one_epoch`:

**For each batch:**

1.  **Train the Discriminator (The "Critic")**
    * Generate a `fake_logits` mask from the Generator and `detach()` it.
    * Calculate `loss_D_real` by showing the Discriminator (image, real_mask) pairs and telling it they are "real" (target=1.0).
    * Calculate `loss_D_fake` by showing the Discriminator (image, fake_logits) pairs and telling it they are "fake" (target=0.0).
    * Sum the losses: `loss_D = (loss_D_real + loss_D_fake) * 0.5`.
    * Perform a backward pass and update **only the Discriminator's weights** (`opt_D.step()`).

2.  **Train the Generator (The "Artist")**
    * **Freeze the Discriminator** (`model_D.requires_grad_(False)`). This is vital; we want to update the Generator *based on* what the (now-fixed) critic thinks.
    * Generate a new `fake_logits` mask (this time, *without* detaching).
    * **Accuracy Goal:** Calculate `loss_G_seg` by comparing `fake_logits` to the `gt_masks` (using the CE+Dice+mSCR combo).
    * **Realism Goal:** Calculate `loss_G_adv` by passing the `fake_logits` through the Discriminator and telling the Generator to *fool* it (target=1.0).
    * Sum the losses: `loss_G = loss_G_seg + (weight * loss_G_adv)`.
    * Perform a backward pass and update **only the Generator's weights** (`opt_G.step()`).

This two-step dance is repeated, and (in theory) both models improve until a Nash equilibrium is reached, where the Generator's masks are "perceptually perfect."

---

## Performance Evaluation Metrics

Loss functions must be differentiable, but evaluation metrics can be exact. We use a separate, more precise set of metrics for validation and testing.

* **Why the split?** The `mSCRLoss` (loss) uses `max_pool2d` as a differentiable *approximation* of morphological dilation. The `compute_mscr_batch` (metric) uses precise, non-differentiable `scipy.ndimage.binary_dilation` operations for a true score.

**Key Metrics Used for Validation & Testing:**

1.  **Pixel-wise Accuracy (`compute_accuracy`)**
    * **What:** Percentage of pixels correctly classified.
    * **Why:** Simple to understand, but *very misleading* on imbalanced datasets. It's included as a low-priority sanity check.

2.  **Mean Intersection over Union (mIoU) (`compute_iou`)**
    * **What:** The standard metric for segmentation. It measures the overlap between the predicted and real masks (`Intersection / Union`).
    * **Why:** A much more robust metric than accuracy, as it heavily penalizes "off-by-one" errors and correctly handles class imbalance.

3.  **mSCR (Mean Shoreline Compliance Rate) (`compute_mscr_batch`)**
    * **This is the project's primary metric for model selection.**
    * **What:** A boundary-specific metric. It calculates the percentage of the *predicted* shoreline that is "close enough" (within a `k`-pixel neighborhood) to the *real* shoreline, and vice-versa.
    * **Why:** This *directly* measures the quality of the boundary. A high mIoU could still have a "wobbly" or blurry edge. A high **mSCR** proves the edge is *precise*. The script tracks this for multiple `k` values (4, 8, 24, 48) to show performance from "very strict" (k=4) to "lenient" (k=48). The `best_val_mscr` is used to save the best model checkpoint.

---

## Dependencies & Quick Start

### Dependencies
This model relies on several key libraries.
```bash
# Installs PyTorch, Albumentations, and Segmentation Models
pip install torch torchvision
pip install albumentations
pip install segmentation-models-pytorch
pip install "numpy<2.0" --force-reinstall # Fixes compatibility with some libraries
pip install opencv-python-headless scipy pandas matplotlib tqdm