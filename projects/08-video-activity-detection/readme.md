# Day 8 — Video Activity Detector

## Project Brief

A video classification system that identifies human activities and suspicious behaviors from surveillance-style video clips.

The project combines **CNN-based spatial feature extraction** with **Bidirectional LSTM temporal modeling** to classify actions across 21 activity categories — ranging from normal daily behaviors to criminal incidents.

The core learning objective for Day 8 is understanding how spatial and temporal information are combined: a CNN reads each frame independently, and an LSTM reads the resulting sequence to understand what is happening across time.

---

## Dataset

**Human Activity & Suspicious Behavior Video Dataset** — 1,334 labeled video clips across 21 classes.

| Split      | Videos |
|------------|-------:|
| Train      | 940    |
| Validation | 194    |
| Test       | 200    |
| **Total**  | **1,334** |

**Normal activities (7):** Clapping, Meeting & Splitting, Sitting, Standing Still, Walking, Walking While Reading, Walking While Using Phone

**Suspicious / criminal activities (14):** Abuse, Arrest, Arson, Assault, Burglary, Explosion, Fighting, Road Accidents, Robbery, Shooting, Shoplifting, Stealing, Vandalism, Normal (surveillance baseline)

The dataset is **imbalanced** — the Normal class has 66 test samples while several minority classes have only 3. This makes accuracy an unreliable standalone metric and requires class-level evaluation.

---

## Pipeline

```
Video Clip
    ↓
Uniform Frame Sampling (16 frames)
    ↓
Resize to 224×224 + Normalize [0, 1]
    ↓
MobileNetV2 (frozen, ImageNet weights)
    ↓
1280-dimensional feature vector per frame
    ↓
Sequence shape: (16, 1280)
    ↓
Bidirectional LSTM(128)
    ↓
Dropout → Dense(64, ReLU) → Dropout
    ↓
Dense(21, Softmax)
    ↓
Activity Class
```

---

## EDA Findings

Video properties vary significantly across the dataset:

- Frame rate: ~30 FPS
- Frame count: ~120–900 frames per clip
- Duration: ~4–30 seconds
- Resolution: inconsistent across clips

Visual inspection of sampled frames confirmed correct loading and label alignment before proceeding to preprocessing.

---

## Preprocessing

### Frame Sampling

Processing every frame of a 30-second clip at 30 FPS would mean 900 frames per video — computationally infeasible on free Colab. Instead, **16 frames are sampled uniformly** across the full duration of each clip.

This converts every video — regardless of length — into a fixed-length sequence, which is required for batched LSTM input.

```
Final shape per video: (16, 224, 224, 3)
```

### CNN Feature Extraction

MobileNetV2 (pretrained on ImageNet, weights frozen) processes each frame independently and produces a 1280-dimensional feature vector via global average pooling.

```python
cnn = MobileNetV2(weights="imagenet", include_top=False, pooling="avg", input_shape=(224, 224, 3))
cnn.trainable = False
```

After extraction, each video is represented as a sequence of frame-level features:

```
X_train → (940,  16, 1280)
X_valid → (194,  16, 1280)
X_test  → (200,  16, 1280)
```

**Why freeze the CNN?** Feature extraction from 940 training videos × 16 frames is already expensive. Fine-tuning MobileNetV2 on top of that would exceed Colab's memory and time limits. The frozen backbone provides strong general visual features from ImageNet that transfer reasonably well to surveillance footage.

---

## Model Architecture

```python
model = Sequential([
    Input(shape=(16, 1280)),
    Bidirectional(LSTM(128)),
    Dropout(0.5),
    Dense(64, activation="relu"),
    Dropout(0.5),
    Dense(21, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
```

**Why BiLSTM instead of LSTM?** A standard LSTM reads the frame sequence left-to-right (frame 1 → frame 16). A Bidirectional LSTM reads it in both directions simultaneously. For activity recognition, the end of an action can provide context for understanding the beginning — BiLSTM captures this.

Early stopping was used to prevent overfitting:

```python
EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)
```

---

## Experiments

### Experiment 1 — Baseline BiLSTM

Single BiLSTM layer with dropout. No class weighting.

| Metric     | Value  |
|------------|--------|
| Test Accuracy | 60.5% |
| Test Loss     | 1.5815 |
| Macro F1      | 0.37   |

This became the final selected configuration.

---

### Experiment 2 — Class Weighting

Added `compute_class_weight("balanced")` to penalize majority class errors more heavily during training.

| Metric     | Value  |
|------------|--------|
| Test Accuracy | 24.0% |

**Result:** Accuracy collapsed. The model shifted its optimization toward minority classes — improving recall on rare classes but at a severe cost to overall accuracy. On an imbalanced test set where the Normal class alone has 66 samples, this trade-off produced a net loss. The class-weighted model was not selected.

---

### Experiment 3 — Stacked BiLSTM

Two BiLSTM layers stacked for deeper temporal modeling:

```
BiLSTM(128, return_sequences=True) → BiLSTM(64) → Dense(64) → Dense(21)
```

| Metric     | Value  |
|------------|--------|
| Test Accuracy | 49.0% |

**Result:** Worse than baseline. Adding depth to the temporal model introduced more parameters without sufficient training data to generalize, leading to overfitting on the training set and weaker test performance.

---

### Experiment 4 — 32 Frames (Incomplete)

Attempted to increase temporal resolution from 16 to 32 frames per video.

**Result:** GPU memory exhaustion during MobileNetV2 feature extraction. Doubling the frames doubles the number of forward passes through the CNN, exceeding available VRAM. The experiment was not completed. NUM_FRAMES = 16 was retained.

---

## Final Results

| Metric        | Value  |
|---------------|--------|
| Test Accuracy | 60.5%  |
| Test Loss     | 1.5815 |
| Macro F1      | 0.37   |

The gap between accuracy (60.5%) and Macro F1 (0.37) reflects the class imbalance. The model performs well on frequent classes and poorly on minority classes with 3 test samples — which destabilizes per-class metrics significantly.

---

## Why Accuracy Alone Is Not Enough

The test set distribution is highly skewed:

```
Normal → 66 test samples
Several classes → 3 test samples
```

A model that always predicts the majority class would achieve misleadingly high accuracy without learning anything useful. Macro F1 averages performance equally across all 21 classes — exposing where the model actually fails.

Evaluation therefore uses: Accuracy, Precision, Recall, F1-score (per class), Macro F1, and Confusion Matrix.

---

## Computational Notes

Feature extraction is the bottleneck. Every video requires 16 forward passes through MobileNetV2. Increasing to 32 frames doubles this cost and exceeded Colab GPU memory limits.

Final configuration chosen as a practical balance:

```
16 frames per video
224 × 224 resolution
MobileNetV2 frozen feature extractor
```

---

## Limitations

- **Class imbalance** — minority classes with 3 test samples produce unstable per-class metrics
- **Frozen CNN** — MobileNetV2 was not fine-tuned for surveillance footage
- **16-frame limit** — fine-grained temporal patterns in longer clips may be lost
- **Compute ceiling** — free Colab GPU constrains frame count, resolution, and model depth

---

## Possible Improvements

- Fine-tune the last few layers of MobileNetV2 on the activity dataset
- Use a stronger backbone (EfficientNet, ResNet50)
- Test alternative frame sampling strategies (motion-aware, keyframe-based)
- Add video-level data augmentation (temporal jitter, horizontal flip)
- Try Transformer-based temporal modeling instead of LSTM
- Collect more samples for minority classes

---

## Technologies

Python · TensorFlow / Keras · OpenCV · NumPy · scikit-learn · Matplotlib · MobileNetV2 · Bidirectional LSTM · GPU (Google Colab)

---

## Day Summary

Day 8 introduced the CNN-LSTM hybrid architecture for video understanding. The key concept is the separation of spatial and temporal reasoning: MobileNetV2 handles what is in each frame, and the BiLSTM handles what is happening across frames over time.

The baseline BiLSTM outperformed both the class-weighted and stacked variants. The most instructive result was Experiment 2 — class weighting shifted the model's optimization in a way that hurt overall performance on an imbalanced test set, demonstrating that solving imbalance requires more than a single training parameter change.

Final test accuracy: **60.5%** | Macro F1: **0.37**
