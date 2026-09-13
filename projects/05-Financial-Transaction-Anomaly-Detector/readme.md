
# Day 5 — Financial Transaction Anomaly Detector

## Overview

An unsupervised anomaly detection pipeline built to identify fraudulent credit card transactions without using any labeled training data. Five detection methods are implemented and compared — from simple statistical rules to machine learning approaches — with full evaluation against ground truth fraud labels held out until the end.

## Dataset

**Credit Card Fraud Detection** — ULB Machine Learning Group (Kaggle)

| Property | Value |
|---|---|
| Total transactions | 284,807 |
| After deduplication | 283,726 |
| Fraud transactions | 473 (0.17%) |
| Features | 30 (Time, V1–V28, Amount) |
| Label column | Class (0 = Normal, 1 = Fraud) |

V1–V28 are PCA-transformed anonymized features from real European cardholder data. `Time` and `Amount` are raw. The `Class` column was used **only for evaluation** — never for training.

## Problem Statement

In real fraud detection systems, labeled fraud data is scarce, delayed, or unavailable. The challenge is to define what "normal" looks like from transaction patterns alone, then flag deviations — without ever seeing a confirmed fraud label during model training.

## Workflow

```
Data Loading → EDA → Preprocessing → Statistical Methods → ML Methods → Comparison → Error Analysis → Documentation
```

## Methods Implemented

### Statistical
| Method | Description |
|---|---|
| IQR | Flags Amount values outside Q1 − 1.5×IQR to Q3 + 1.5×IQR |
| Z-Score | Flags Amount values more than 3 standard deviations from the mean |

### Machine Learning
| Method | Description |
|---|---|
| Isolation Forest | Randomly isolates points; anomalies require fewer splits |
| LOF | Compares local density of each point to its k nearest neighbors |
| Elliptic Envelope | Fits a multivariate Gaussian; flags points with high Mahalanobis distance |

## Results

| Method | Precision | Recall | F1 |
|---|---|---|---|
| IQR (Amount only) | 0.27% | 18.39% | 0.54% |
| Z-Score (Amount only) | 0.27% | 2.33% | 0.49% |
| Isolation Forest | 13.04% | 39.11% | 19.56% |
| LOF | 0.21% | 0.63% | 0.32% |
| **Elliptic Envelope** | **26.50%** | **79.49%** | **39.75%** |

**Best model: Elliptic Envelope** — correctly identified 79.49% of all fraud transactions with no labels during training.

## Key Findings

**1. Accuracy is meaningless on imbalanced data.**
A model predicting Normal for every transaction achieves 99.83% accuracy while catching zero fraud. Precision, Recall, and F1 are the only meaningful metrics.

**2. Single-feature statistical methods cannot detect fraud.**
IQR and Z-Score examine Amount in isolation. Fraud is defined by unusual combinations across all 30 features simultaneously — no single feature captures it reliably.

**3. LOF is memory-prohibitive on large datasets.**
LOF requires pairwise distance computation (O(n²) memory). On 283,726 rows, this exceeds Colab's free tier RAM limit (~12 GB). The kernel crashes silently mid-execution. Fix: run LOF on a representative 10,000–15,000 row sample, which is statistically valid for a density estimator.

**4. Elliptic Envelope suits PCA-structured features.**
V1–V28 are PCA-transformed outputs, which tend toward Gaussian distributions — exactly what Elliptic Envelope assumes. Matching the method to the data's structure matters more than blindly testing every algorithm.

**5. Contamination is a business decision.**
The contamination parameter controls the precision/recall trade-off. The optimal value depends on how costly false positives (blocking legitimate customers) are versus false negatives (missing real fraud) — a business decision, not a purely technical one.

## Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core language |
| pandas / NumPy | Data loading and manipulation |
| scikit-learn | IsolationForest, LOF, EllipticEnvelope, StandardScaler |
| matplotlib / seaborn | Visualizations |
| Google Colab | Execution environment |

## Project Structure

```
day-05-anomaly-detector/
│
├── DAY05_Financial_Transaction_Anomaly_Detector.ipynb   # Main notebook
├── README.md                                            # This file
└── data/
    └── creditcard.csv                                   # Source: Kaggle 
```

## How to Run

1. Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
2. Upload to Google Drive at `MyDrive/data/creditcard.csv`
3. Open the notebook in Google Colab
4. Run all cells in order

> **Note:** Do not run the LOF cell on the full dataset in Colab — it will exhaust available RAM. Use a sample of 10,000–15,000 rows instead.

---

*Part of the 10-Day Advanced ML Practical Milestone*
*Day 4 ← [Day 5] → Day 6*
