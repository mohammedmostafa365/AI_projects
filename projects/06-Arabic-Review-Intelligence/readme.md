# Day 6 — Arabic Review Intelligence

> Part of my 10-day Advanced AI Practical Milestone

## Overview

A sentiment classifier for Arabic text reviews built using classical NLP techniques.
The pipeline covers Arabic-specific text preprocessing, two vectorization strategies,
and three classifiers — compared systematically to find the best combination.

**Task:** 3-class sentiment classification — Positive / Negative / Mixed  
**Dataset:** 100,000 Arabic reviews, perfectly balanced across all three classes  
**Source:** [Arabic Reviews Dataset — Kaggle](https://www.kaggle.com/datasets/abedkhooli/arabic-100k-reviews)

---

## Results

| Model | Accuracy | Mixed F1 | Negative F1 | Positive F1 |
|---|---|---|---|---|
| Naive Bayes (BoW) | 0.61 | 0.50 | 0.67 | 0.66 |
| Logistic Regression (BoW) | 0.62 | 0.53 | 0.67 | 0.66 |
| LinearSVC (BoW) | 0.61 | 0.52 | 0.67 | 0.66 |
| Logistic Regression (TF-IDF) | 0.64 | 0.55 | 0.70 | 0.68 |
| LinearSVC (TF-IDF) | 0.63 | 0.53 | 0.68 | 0.67 |
| **LinearSVC (TF-IDF + Bigrams)** | **0.65** | **0.55** | **0.69** | **0.69** |

**Best model:** LinearSVC with TF-IDF + Bigrams — **65% accuracy**

---

## Pipeline

```
Raw Arabic Text
      ↓
Diacritic Removal       — strip تشكيل marks that create false token variants
Alef Normalization      — unify أ إ آ ا into a single form
Non-Arabic Removal      — strip punctuation, numbers, English characters
Whitespace Cleanup      — collapse multiple spaces
Tokenization            — split on whitespace
Stopword Removal        — remove في، من، على، هذا and other zero-signal words
Stemming (ISRIStemmer)  — reduce words to common root form
      ↓
Vectorization (BoW → TF-IDF → TF-IDF + Bigrams)
      ↓
Classification (Naive Bayes / Logistic Regression / LinearSVC)
      ↓
Evaluation + Error Analysis
```

---

## Key Findings

**Mixed class is the hardest.** F1 stays around 0.55 across all models. Mixed reviews contain both positive and negative language — a bag-of-words model has no way to weigh the balance, it only counts words. This is a known ceiling for classical NLP methods.

**TF-IDF consistently beats BoW.** Switching from raw counts to TF-IDF improved accuracy by ~2% across all models by down-weighting high-frequency but low-signal words.

**Bigrams help modestly.** Adding two-word sequences captures some negation and phrase context (65% vs 63%), but the gain is limited because stemming already destroys word order.

**Variable shadowing is a critical bug.** Overwriting `df['text']` across multiple notebook cells caused the classifier to receive only one class. Fixed by always building a separate `clean_text` column and never mutating the original.

---

## Stack

- Python, pandas, NumPy
- NLTK — Arabic stopwords, ISRIStemmer
- scikit-learn — CountVectorizer, TfidfVectorizer, MultinomialNB, LogisticRegression, LinearSVC
- matplotlib

---

## What's Next

Day 9 — Arabic Document Intelligence using Transformers (AraBERT / CAMeL-BERT).
Contextual embeddings should directly address the Mixed class ceiling that classical methods hit here.
