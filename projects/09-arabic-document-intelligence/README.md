# Day 9 — Arabic Sentiment Classification with Transformers

## Project Overview

Built an Arabic sentiment classification system using Egyptian customer reviews.

The task was a 3-class classification problem:

* Negative
* Neutral
* Positive

The dataset contained 40,046 reviews originally. One review with a missing text value was removed, leaving 40,045 usable samples.

The dataset was imbalanced:

* Positive: 59.73%
* Negative: 35.46%
* Neutral: 4.81%

Because of this imbalance, Macro F1 was used alongside accuracy as the main evaluation metric.

## Data Preparation

The dataset was split using stratified sampling:

* 80% Training
* 10% Validation
* 10% Testing

Labels were mapped to numerical values:

```text
0 → Negative
1 → Neutral
2 → Positive
```

For the Transformer model, Arabic text was tokenized using the CAMeLBERT tokenizer with a maximum sequence length of 128 tokens.

Dynamic padding was used with `DataCollatorWithPadding` to reduce unnecessary padding and improve training efficiency.

## Model 1 — CAMeLBERT

Used:

`CAMeL-Lab/bert-base-arabic-camelbert-mix`

The pretrained model was fine-tuned for the 3-class sentiment classification task.

Training configuration:

* Epochs: 2
* Learning rate: 2e-5
* Training batch size: 16
* Validation batch size: 32
* Weight decay: 0.01
* GPU: NVIDIA Tesla T4

### Test Results

```text
Accuracy: 86.69%
Macro F1: 0.633
```

Per-class F1:

```text
Negative: 0.85
Neutral:  0.13
Positive: 0.91
```

The model performed well on positive and negative reviews but struggled with the neutral class.

The confusion matrix showed that most neutral reviews were classified as either negative or positive.

Further inspection showed that the neutral class contains many ambiguous, noisy, or potentially mislabeled examples. Some reviews labeled as neutral expressed clear dissatisfaction, while others were questions, requests, corrupted text, or mixed-sentiment statements.

## Model 2 — TF-IDF + Logistic Regression

A classical NLP baseline was built using:

```text
TF-IDF
   ↓
Logistic Regression
```

TF-IDF was fitted only on the training data to avoid data leakage.

Configuration:

* N-grams: 1–2
* Maximum features: 50,000
* Classifier: Logistic Regression

### Test Results

```text
Accuracy: 84.0%
Macro F1: 0.57
```

Per-class F1:

```text
Negative: 0.81
Neutral:  0.02
Positive: 0.88
```

The baseline almost completely failed to identify the neutral class, correctly classifying only 2 out of 193 neutral test examples.

## Model Comparison

| Model                        | Accuracy | Macro F1 | Negative F1 | Neutral F1 | Positive F1 |
| ---------------------------- | -------: | -------: | ----------: | ---------: | ----------: |
| TF-IDF + Logistic Regression |    84.0% |     0.57 |        0.81 |       0.02 |        0.88 |
| CAMeLBERT                    |   86.69% |    0.633 |        0.85 |       0.13 |        0.91 |

## Key Findings

CAMeLBERT outperformed the classical TF-IDF + Logistic Regression baseline on both accuracy and Macro F1.

The biggest improvement was in the neutral class:

```text
TF-IDF + Logistic Regression → 0.02 F1
CAMeLBERT                  → 0.13 F1
```

This demonstrates the advantage of contextual Transformer representations over traditional bag-of-words-style TF-IDF features for Arabic text classification.

However, the neutral class remains difficult because it represents only 4.81% of the dataset and contains several ambiguous or noisy examples.

## What I Learned

* How to use Hugging Face Transformers for Arabic NLP.
* How tokenization converts text into model inputs.
* How attention masks work.
* How to fine-tune a pretrained Transformer for classification.
* How dynamic padding improves training efficiency.
* How to use Hugging Face `Trainer`.
* Why Macro F1 is important for imbalanced classification.
* How to perform error analysis using a confusion matrix and misclassified examples.
* How to establish a classical TF-IDF + Logistic Regression baseline.
* How to compare a Transformer model against a traditional NLP approach.

## Final Conclusion

The project showed that a fine-tuned Arabic Transformer can outperform a traditional TF-IDF + Logistic Regression approach on this sentiment classification task.

CAMeLBERT achieved 86.69% test accuracy and a 0.633 Macro F1, compared with 84.0% accuracy and 0.57 Macro F1 for the TF-IDF baseline.

The main remaining challenge is the neutral class, where both models performed poorly. Error analysis suggests that class imbalance and noisy or ambiguous labels are major factors.

This provides a clear direction for future improvement through better handling of class imbalance, improved data quality, and further Transformer fine-tuning.

