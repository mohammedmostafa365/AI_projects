
# Day 7 — Smart Traffic Vision

> **AI Advanced Ahmed Yousrey Course — 10-Day Practical Milestone**

---

## Overview

A deep learning pipeline for classifying traffic signs using the **GTSRB (German Traffic Sign Recognition Benchmark)** dataset — 43 classes and ~39,000 training images.

The project covers the full CNN workflow from raw image loading to transfer learning fine-tuning, with a direct comparison between a custom-built CNN and a pretrained MobileNetV2 model.

---

## Objectives

- Build and train a **CNN from scratch** using TensorFlow/Keras
- Apply **image preprocessing** and **data augmentation** for generalization
- Implement **transfer learning** with MobileNetV2 (feature extraction + fine-tuning)
- Evaluate and compare both models with accuracy curves and a confusion matrix
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
Transfer Learning — MobileNetV2 (Phase 1: frozen → Phase 2: fine-tune)
    ↓
Evaluation — accuracy curves, confusion matrix, error analysis
    ↓
Save Models
```

---

## Models

### CNN from Scratch

| Layer Block | Details |
|---|---|
| Block 1 | Conv2D(32) × 2 → BatchNorm → MaxPool → Dropout(0.25) |
| Block 2 | Conv2D(64) × 2 → BatchNorm → MaxPool → Dropout(0.25) |
| Block 3 | Conv2D(128) → BatchNorm → MaxPool → Dropout(0.3) |
| Head | Flatten → Dense(256) → BatchNorm → Dropout(0.5) → Dense(43, softmax) |

### Transfer Learning — MobileNetV2

| Phase | Description |
|---|---|
| Phase 1 | Base frozen, only classification head trained (LR = 1e-3) |
| Phase 2 | Top 30 base layers unfrozen, full fine-tuning (LR = 1e-5) |

---

## Results

| Model | Best Validation Accuracy |
|---|---|
| CNN from Scratch | ~88–92% |
| MobileNetV2 — Feature Extraction | ~93–96% |
| MobileNetV2 — Fine-Tuned | ~96–98% |

---

## Key Takeaways

1. **Transfer learning dominates** — MobileNetV2's ImageNet features generalize well to traffic signs, reaching higher accuracy faster than any custom CNN.

2. **Two-phase fine-tuning is essential** — Training the head first stabilizes weights before unfreezing the backbone. Skipping this risks destroying pretrained features.

3. **Class imbalance must be addressed** — Without class weights, the model ignores minority classes. Weighting ensures every class gets equal learning pressure.

4. **Augmentation must be domain-aware** — Horizontal flipping was intentionally excluded because it changes the semantic meaning of directional signs.

5. **CNNs learn hierarchically** — Shallow layers detect edges, middle layers detect shapes, deep layers detect full sign structures. This hierarchy is why transfer learning works.

---

## Tech Stack

| Tool | Purpose |
|---|---|
| TensorFlow / Keras | Model building and training |
| MobileNetV2 | Pretrained backbone for transfer learning |
| ImageDataGenerator | Preprocessing and augmentation pipeline |
| PIL / Matplotlib | Image loading and visualization |
| scikit-learn | Classification report and confusion matrix |
| Kaggle | Dataset source and GPU runtime |

---

## How to Run

1. Open the notebook in **Kaggle** (GPU accelerator recommended)
2. Attach the GTSRB dataset: `meowmeowmeowmeowmeow/gtsrb-german-traffic-sign`
3. Run all cells in order
4. Models are saved as `.keras` files at the end

---

*Day 7 of 10 — AI Advanced Practical Milestone*
