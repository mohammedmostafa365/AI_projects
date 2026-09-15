# Day 7 — Smart Traffic Vision

> **AI Advanced Ahmed Yousrey Course — 10-Day Practical Milestone**

---

## Overview

A deep learning pipeline for classifying traffic signs using the **GTSRB (German Traffic Sign Recognition Benchmark)** dataset — 43 classes and ~39,000 training images.

The project covers the full CNN workflow from raw image loading to evaluation, with a custom-built CNN trained from scratch and analyzed with per-class metrics, a confusion matrix, and error analysis.

---

## Objectives

- Build and train a **CNN from scratch** using TensorFlow/Keras
- Apply **image preprocessing** and **data augmentation** for generalization
- Handle **class imbalance** with class weights during training
- Evaluate the model with accuracy curves and a confusion matrix
- Perform **error analysis** on misclassified samples

---

## Dataset

| Property | Value |
|---|---|
| Name | GTSRB — German Traffic Sign Recognition Benchmark |
| Source | Kaggle |
| Classes | 43 |
| Training images | ~39,000 |
| Image type | Real-world photos (varying size, lighting, occlusion) |
| Format | Class folders under `Train/` |

**Class imbalance:** Some classes have 10× more samples than others. Handled via class weights during training.

---

## Project Structure

```
day07-smart-traffic-vision/
├── day07-smart-traffic-vision.ipynb   # Main notebook
└── README.md
```

---

## Workflow

```
Load Dataset
    ↓
EDA — class distribution, sample images, size analysis
    ↓
Preprocessing — resize to 64×64, normalize to [0, 1]
    ↓
Augmentation — rotation, zoom, shift, brightness (training only)
    ↓
CNN from Scratch — 3 conv blocks, BatchNorm, Dropout
    ↓
Evaluation — accuracy curves, confusion matrix, error analysis
    ↓
Save Models
```

---

## Model — CNN from Scratch

| Layer Block | Details |
|---|---|
| Block 1 | Conv2D(32) × 2 → BatchNorm → MaxPool → Dropout(0.25) |
| Block 2 | Conv2D(64) × 2 → BatchNorm → MaxPool → Dropout(0.25) |
| Block 3 | Conv2D(128) → BatchNorm → MaxPool → Dropout(0.3) |
| Head | Flatten → Dense(256) → BatchNorm → Dropout(0.5) → Dense(43, softmax) |

**Training setup:** Adam (LR = 1e-3) + categorical cross-entropy, 30 epochs, class weights, EarlyStopping (patience = 5, restores best weights) and ReduceLROnPlateau (factor = 0.5).

---

## Results

| Model | Best Validation Accuracy |
|---|---|
| CNN from Scratch | ~88–92% |

---

## Key Takeaways

1. **A scratch CNN is a strong baseline** — With only 3 conv blocks plus BatchNorm and Dropout, the model reaches ~88–92% validation accuracy on GTSRB, proving that a well-regularized CNN can learn traffic-sign recognition from raw pixels alone.

2. **Regularization controls overfitting** — Tracking the gap between training and validation accuracy shows where the model starts memorizing; BatchNorm, Dropout, and early stopping keep that gap under control.

3. **Class imbalance must be addressed** — Without class weights, the model ignores minority classes. Weighting ensures every class gets equal learning pressure.

4. **Augmentation must be domain-aware** — Horizontal flipping was intentionally excluded because it changes the semantic meaning of directional signs.

5. **CNNs learn hierarchically** — Shallow layers detect edges, middle layers detect shapes, deep layers detect full sign structures.

---

## Tech Stack

| Tool | Purpose |
|---|---|
| TensorFlow / Keras | Model building and training |
| ImageDataGenerator | Preprocessing and augmentation pipeline |
| PIL / Matplotlib | Image loading and visualization |
| scikit-learn | Classification report and confusion matrix |
| Kaggle | Dataset source and GPU runtime |

---

## How to Run

1. Open the notebook in **Kaggle** (GPU accelerator recommended)
2. Attach the GTSRB dataset: `meowmeowmeowmeowmeow/gtsrb-german-traffic-sign`
3. Run all cells in order
4. The model is saved as a `.keras` file at the end

---

*Day 7 of 10 — AI Advanced Practical Milestone*
