
# Video Activity Detector

A video activity classification system that combines **CNN-based spatial feature extraction** with a **Bidirectional LSTM (BiLSTM)** to recognize human activities from video sequences.

The project focuses on learning both:

* **Spatial information** from individual video frames using MobileNetV2.
* **Temporal information** across frames using a Bidirectional LSTM.

---

## Project Pipeline

```text
Video
  ↓
Frame Sampling
  ↓
Resize + Normalize
  ↓
MobileNetV2
  ↓
1280-D Spatial Features
  ↓
BiLSTM
  ↓
Dense Layers
  ↓
Softmax
  ↓
Activity Class
```

---

## Dataset

The project uses a video activity classification dataset containing **21 activity classes**.

### Classes

```text
0  Vandalism
1  Stealing
2  Shoplifting
3  Shooting
4  Robbery
5  Roadaccidents
6  Normal
7  Walking
8  Walking_While_Using_Phone
9  Walking_While_Reading_Book
10 Standing_Still
11 Sitting
12 Fighting
13 Explosion
14 Meet_and_Split
15 Burglary
16 Clapping
17 Assault
18 Arson
19 Arrest
20 Abuse
```

### Dataset Split

| Split      |   Videos |
| ---------- | -------: |
| Train      |      940 |
| Validation |      194 |
| Test       |      200 |
| **Total**  | **1334** |

The dataset is imbalanced. For example, the test set contains many more samples for the `Normal` class than for several other activities.

Therefore, accuracy is not considered sufficient by itself, and class-level metrics such as precision, recall, and F1-score are also examined.

---

## Exploratory Data Analysis

Before processing the videos, several dataset properties were investigated.

### Video Properties

* Frame rate: approximately **30 FPS** across the dataset.
* Frame count: approximately **120–900 frames**.
* Duration: approximately **4–30 seconds**.
* Video resolutions vary across the dataset.

The videos were also visually inspected using sampled frames to verify that the data was being loaded correctly.

---

# Preprocessing

## Frame Sampling

Instead of processing every frame of a video, a fixed number of frames is sampled uniformly throughout the video.

The final configuration uses:

```python
NUM_FRAMES = 16
```

Uniform sampling allows videos with different durations to be converted into sequences with the same temporal length.

Each video therefore becomes:

```text
16 frames
```

---

## Image Processing

Each sampled frame is:

1. Converted from BGR to RGB.
2. Resized to:

```text
224 × 224
```

3. Normalized to the range:

```text
[0, 1]
```

The resulting shape for one video is:

```text
(16, 224, 224, 3)
```

---

# CNN Feature Extraction

Instead of training a CNN from scratch, **MobileNetV2 pretrained on ImageNet** is used as a feature extractor.

```python
cnn = MobileNetV2(
    weights="imagenet",
    include_top=False,
    pooling="avg",
    input_shape=(224, 224, 3)
)

cnn.trainable = False
```

The CNN is frozen, meaning its weights are not updated during training.

After processing each frame, MobileNetV2 produces a:

```text
1280-dimensional feature vector
```

Therefore, each video becomes:

```text
16 × 1280
```

and the complete datasets have the following shapes:

```text
X_train → (940, 16, 1280)
X_valid → (194, 16, 1280)
X_test  → (200, 16, 1280)
```

---

# Temporal Modeling

The extracted CNN features are passed to a Bidirectional LSTM.

The baseline architecture is:

```text
16 × 1280
    ↓
BiLSTM(128)
    ↓
Dropout(0.5)
    ↓
Dense(64, ReLU)
    ↓
Dropout(0.5)
    ↓
Dense(21, Softmax)
```

### Model

```python
model = models.Sequential([
    layers.Input(shape=(16, 1280)),

    layers.Bidirectional(
        layers.LSTM(128)
    ),

    layers.Dropout(0.5),

    layers.Dense(
        64,
        activation="relu"
    ),

    layers.Dropout(0.5),

    layers.Dense(
        21,
        activation="softmax"
    )
])
```

### Compilation

```python
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
```

Early stopping was used during training to reduce unnecessary overfitting:

```python
EarlyStopping(
    monitor="val_loss",
    patience=4,
    restore_best_weights=True
)
```

---

# Experiments

Several experiments were performed to investigate possible improvements.

## Experiment 1 — Baseline BiLSTM

The first model used a single Bidirectional LSTM with dropout.

### Result

```text
Test Accuracy: 60.5%
Test Loss:     1.5815
Macro F1:      0.37
```

The baseline model achieved the best test accuracy among the completed experiments.

---

## Experiment 2 — Class Weighting

Because the dataset is imbalanced, class weights were calculated using:

```python
compute_class_weight(
    class_weight="balanced"
)
```

The model was then trained using the calculated class weights.

### Result

```text
Test Accuracy: 24.0%
```

The class-weighting approach significantly reduced the overall test accuracy in this experiment.

Therefore, the baseline configuration was retained.

---

## Experiment 3 — Stacked BiLSTM

A deeper temporal model was tested using two Bidirectional LSTM layers:

```text
16 × 1280
    ↓
BiLSTM(128, return_sequences=True)
    ↓
BiLSTM(64)
    ↓
Dense(64)
    ↓
Dense(21)
```

### Result

```text
Test Accuracy: 49.0%
```

This was lower than the baseline model.

Therefore, increasing the temporal model depth did not improve performance in this experiment.

---

## Experiment 4 — Increasing the Number of Frames

The number of sampled frames was increased:

```text
16 → 32 frames
```

The goal was to provide the temporal model with more information from each video.

However, feature extraction with MobileNetV2 caused GPU memory exhaustion.

For example, processing multiple videos simultaneously resulted in CUDA/ GPU out-of-memory errors.

The feature extraction batch size had to be reduced significantly.

Because of the available computational resources and the long processing time, the experiment was not completed.

The project therefore retained:

```text
NUM_FRAMES = 16
```

This also provides a practical compromise between temporal information, memory usage, and processing time.

---

# Final Model

Based on the completed experiments, the final selected configuration is the baseline BiLSTM model:

```text
16 sampled frames
        ↓
MobileNetV2
        ↓
16 × 1280 features
        ↓
BiLSTM(128)
        ↓
Dense(64)
        ↓
Softmax(21 classes)
```

### Final Test Performance

```text
Accuracy: 60.5%
Macro F1: 0.37
```

The Macro F1 score is substantially lower than the accuracy, which is expected to be influenced by the strong class imbalance in the test set.

Several minority classes had only a small number of test samples, making their individual metrics relatively unstable.

---

# Why Accuracy Alone Is Not Enough

The test set is imbalanced.

For example:

```text
Normal → 66 test samples
```

while several classes contain only:

```text
3 test samples
```

As a result, a model can achieve reasonable overall accuracy while still performing poorly on some minority classes.

For this reason, the project also evaluates:

* Precision
* Recall
* F1-score
* Macro F1
* Confusion Matrix

The confusion matrix is particularly useful for identifying activities that are frequently confused with each other.

---

# Computational Considerations

The CNN feature extraction stage is computationally expensive because every sampled frame must pass through MobileNetV2.

Increasing the number of frames from 16 to 32 approximately doubles the number of frames processed per video.

This significantly increases:

* GPU memory usage
* Feature extraction time
* Overall processing time

GPU memory limitations were therefore considered when selecting the final preprocessing configuration.

The final system uses:

```text
16 frames/video
224 × 224 resolution
MobileNetV2 feature extraction
```

as a practical configuration for the available computational resources.

---

# Limitations

The current system has several limitations:

1. **Dataset imbalance**

   Some activities have significantly fewer samples than others.

2. **Limited temporal sampling**

   Only 16 frames are used from each video, so some fine-grained temporal information may be lost.

3. **Frozen CNN**

   MobileNetV2 is used as a fixed feature extractor rather than being fine-tuned for the activity dataset.

4. **Computational constraints**

   Processing more frames significantly increases GPU memory requirements and processing time.

5. **Minority-class performance**

   Some classes have very few test samples, resulting in unstable per-class metrics.

---

# Possible Future Improvements

Future experiments could include:

* Fine-tuning the last layers of MobileNetV2.
* Using a stronger pretrained CNN backbone.
* Testing different frame sampling strategies.
* Increasing the number of frames using more memory-efficient feature extraction.
* Data augmentation.
* More advanced temporal architectures such as Transformer-based temporal models.
* Better handling of class imbalance.
* Collecting or using more samples for minority classes.
* Hyperparameter tuning for the BiLSTM architecture.

---

# Technologies

* Python
* TensorFlow / Keras
* OpenCV
* NumPy
* Scikit-learn
* Matplotlib
* MobileNetV2
* Bidirectional LSTM
* GPU acceleration

---

# Conclusion

This project demonstrates a complete video classification pipeline that combines pretrained CNN-based spatial feature extraction with recurrent temporal modeling.

The final baseline configuration achieved:

```text
Test Accuracy: 60.5%
Macro F1:      0.37
```

Several alternative configurations were experimentally evaluated. The baseline BiLSTM performed better than the tested class-weighted and stacked-BiLSTM approaches.

The final configuration was selected while considering both model performance and available computational resources.
