# CIFAR-10 Image Classification — ANN vs CNN

This notebook is part of my Week 4 deep learning assignment. The goal was to build image classification models on the CIFAR-10 dataset and compare how ANN and CNN perform on the same task.

---

## What's in the notebook

- Loading and exploring the CIFAR-10 dataset (class distribution, pixel intensity, sample images)
- Preprocessing — normalization and flattening for ANN
- Building and training an ANN model
- Building and training a CNN model
- Using Data Augmentation for Training models
- Comparing all three models on accuracy and loss
- Confusion matrix and per-class breakdown on the best model
- Misclassified sample visualization

---

## Dataset

CIFAR-10 — 60,000 color images (32×32×3) across 10 classes:
airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

- 50,000 training images
- 10,000 test images
- Perfectly balanced (5000 per class)

---

## Models

**ANN** — flattens the image into a 3072-length vector and passes it through dense layers. Simple but loses all spatial information, which limits how well it can do on image data.

**CNN** — uses convolutional layers to detect local patterns like edges and textures. Includes batch normalization, dropout, and EarlyStopping + ReduceLROnPlateau callbacks.

**CNN + Augmentation** — same architecture as CNN but with random flips, rotations, zoom, and translation applied during training to reduce overfitting.

---

## Results

| Model | Test Accuracy |
|-------|--------------|
| ANN | 41.21% |
| CNN | 80.99% |
| CNN + Augmentation | 71.87% |

The augmented CNN scored lower than the plain CNN here because 20 epochs isn't enough for augmented training to converge — it was still improving when EarlyStopping cut it off. With 35–40 epochs it should outperform the base CNN.

---

## Requirements

```
tensorflow
numpy
pandas
matplotlib
seaborn
scikit-learn
```

---

## How to run

Just open the notebook in Jupyter or Google Colab. 

install this dependancies using pip

```
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn
```
and run all cells in order. CIFAR-10 downloads automatically via Keras on the first run.
