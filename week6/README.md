# Week 6 – Denoising Autoencoder on MNIST

Built a convolutional autoencoder that learns to remove Gaussian noise from MNIST digit images. The model takes a noisy image as input and tries to reconstruct the clean version using an encoder-decoder architecture.

---

## What it does

- Loads MNIST and adds Gaussian noise (noise factor = 0.4)
- Trains a Conv autoencoder to map noisy → clean images
- Evaluates reconstruction quality overall and per digit class
- Tests how the model holds up at different noise levels (0.1 to 0.8)

---

## Model

Simple encoder-decoder setup:

- **Encoder** – two Conv2D + MaxPooling blocks, compresses 28×28×1 down to 7×7×64
- **Decoder** – mirrors the encoder using UpSampling2D, ends with sigmoid activation
- **Loss** – MSE (pixel-level reconstruction)
- **Optimizer** – Adam

---

## Results

- Test MSE: **0.11396**
- Digit 1 was the easiest to reconstruct (simple vertical stroke)
- Digit 0 had the highest error (oval shape is harder to recover after noise)
- Reconstruction quality stayed fairly flat across noise levels 0.1–0.8, likely because the model converged early

---

## Files

| File | Description |
|------|-------------|
| `week6-mnist.ipynb` | Main notebook with all code, plots, and analysis |
| `denoising_autoencoder_mnist.keras` | Saved model weights |

---

## Libraries used

- TensorFlow / Keras
- NumPy
- Matplotlib

---

## Notes

Ran on Kaggle with dual T4 GPUs using `MirroredStrategy`. Training stopped early around epoch 10 — the val loss flattened pretty quickly so there wasn't much point letting it run longer.