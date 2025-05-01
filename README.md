# Wavelet-AR: Multiscale Autoregressive Image Modeling using Wavelet-VQVAE

This project explores a multiscale generative modeling approach combining discrete wavelet transforms (DWT), VQ-VAE quantization for each frequency band, and a Transformer for autoregressive synthesis.

---

## Project Overview

Traditional generative models often struggle with maintaining fidelity at coarse scales. We propose a hybrid method that:

- Decomposes images into wavelet subbands (LL, LH, HL, HH),
- Trains a VQ-VAE on each subband for discrete latent representations,
- Uses a Transformer to autoregressively model the sequence of discrete codes.

This results in faster training and multiresolution image generation capabilities.

---

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

While cloning from GitHub, 

You can use this command - git clone --branch standard-transformer https://github.com/Electrum9/LaplacianVQVAE-AR.git

---

## Reproducing Results

### Step 1: Train Wavelet VQ-VAEs

Use one of the provided VQ-VAE variants:
- `vqvae.py` (default)
- `vqvae-l1.py`, `vqvae-l2.py`, or no-SSIM variants

```bash
python vqvae.py
```

- Trains VQ-VAEs for each wavelet band using FFHQ-256
- Saves checkpoint to `wavelet_vqvae.pth`

### Step 2: Train the Transformer

Use one of the Transformer variants:
- `standard_transformer.py` (baseline)
- `transformer.py` (Wavelet-AR)
- `compare-transformers.py` (benchmarking multiple)

```bash
python transformer.py
```

- Loads the VQ-VAE checkpoint
- Autoregressively models the wavelet-index sequences
- Saves model to `wavelet_transformer.pth`

### Step 3: Generate Samples + Reconstruct

```bash
python wavelet-recon.py
```

- Generates image samples from latent indices
- Uses inverse wavelet transform to reconstruct full images
- Saves results to `generated_samples_X.png`

### Step 4: Inference Only

```bash
python run_inference.py
```

---

## Logging & Evaluation

- TensorBoard logs are saved in `runs/`
- Use `plot-loss.py` to visualize training curves
- Metrics computed include L1, SSIM

Launch TensorBoard:
```bash
tensorboard --logdir=runs
```

---

## Directory Structure

```
.
├── compare-transformers.py     # Compare Wavelet-AR vs Standard
├── generated_samples_*.png     # Sample outputs
├── plot-loss.py                # Plot training loss
├── run_inference.py            # Sampling interface
├── standard_transformer.py     # Baseline transformer model
├── transformer.py              # Wavelet-AR transformer
├── vqvae*.py                   # VQ-VAE variants (l1, l2, no-SSIM)
├── wavelet-recon.py            # Generation and reconstruction script
├── wavelet_and_recon.png       # Visualization of subbands and recon
├── wavelet_transformer*.pth    # Saved Transformer weights
├── requirements.txt
├── README.md
```

---

## Dataset

- [FFHQ-256](https://huggingface.co/datasets/merkol/ffhq-256) is used via 🤗 `datasets`

It is automatically downloaded using:

```python
load_dataset("merkol/ffhq-256", split="train")
```

---

## Authors

- Vikram Bhagavatula
- Ritvika Sonawane
- Siddharth Shah

Department of Electrical and Computer Engineering  
Carnegie Mellon University
