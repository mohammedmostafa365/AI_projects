# Day 3 — Job Application Intelligence

## Project Overview

The goal of this project was to build a machine learning system that predicts whether a job application is a good match for a specific job.

The system uses information from:

* Resume
* Job Role
* Job Description
* Age
* Gender

The target variable is `Best Match`, a binary classification label.

The main focus of this project was applying classical NLP techniques to a real-world style job-matching problem, building clean pipelines, comparing models systematically with cross-validation, and tuning the best model with GridSearchCV.

---

## Data Preparation

Before modeling, the dataset was inspected and cleaned.

Key preprocessing steps included:

* Removed unnecessary identifying information such as `Applicant Name`.
* Removed `Race` and `Ethnicity` because they are not required for job matching and may introduce unwanted bias.
* Converted `Gender` into a numerical feature.
* Verified missing values and duplicates — none found.
* Split the data into training and testing sets before learning any preprocessing parameters.

The dataset was relatively balanced:

```text
Class 0: 51.5%
Class 1: 48.5%
```

Therefore, predicting the majority class would give a baseline accuracy of approximately 51.5%.

---

## NLP Feature Engineering

Instead of treating the text as ordinary categorical data, TF-IDF was used to convert the text into numerical features.

Separate TF-IDF vectorizers were used for:

* `Resume`
* `Job Description`
* `Job Roles`

This was intentional because each text source has a different meaning:

```text
Resume          → Skills and experience
Job Description → Requirements and responsibilities
Job Role        → Position/title information
```

The resulting sparse matrices were combined using `hstack`.

The numerical features `Age` and `Gender` were then added to the sparse text representation.

---

## Important NLP Concept: Fit vs Transform

One of the most important concepts reinforced in this project was the difference between:

```text
fit()
transform()
fit_transform()
```

The TF-IDF vectorizer must learn its vocabulary and IDF values from the training data only.

Therefore `fit_transform()` was called only on training data, and `transform()` only on test data. Using `fit_transform()` on the test set would allow information from the test data to influence preprocessing and introduce data leakage.

---

## Building with Pipelines

In the first iteration, preprocessing steps were written manually across many cells — fitting each TF-IDF separately, stacking matrices by hand, then passing the result to the model.

The project was then rebuilt using `Pipeline` and `ColumnTransformer` from scikit-learn.

`ColumnTransformer` applies different transformations to different columns and combines the output automatically:

```python
preprocessor = ColumnTransformer(transformers=[
    ("resume",   TfidfVectorizer(), "Resume"),
    ("job_desc", TfidfVectorizer(), "Job Description"),
    ("job_role", TfidfVectorizer(), "Job Roles"),
    ("numeric",  "passthrough",     ["Age", "Gender"])
])

pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier",   LogisticRegression(max_iter=1000))
])
```

The full workflow — preprocessing and training — is now a single `.fit()` call. The pipeline enforces correct data flow automatically and eliminates any risk of accidental leakage.

---

## Model Comparison with Cross-Validation

In the first iteration, each model was evaluated on a single train/test split. A single split may be lucky or unlucky depending on which samples end up in the test set.

5-fold cross-validation was used instead. The data is split into 5 parts. The model trains on 4 parts and tests on the remaining 1, repeated 5 times. The mean and standard deviation of the 5 scores give a more honest and stable performance estimate.

### Initial Models — Cross-Validation Results

| Model                | Mean Accuracy | Std  |
|----------------------|:-------------:|:----:|
| Logistic Regression  | ~0.63         | low  |
| Linear SVC           | ~0.63         | low  |
| Naive Bayes          | ~0.55         | low  |

All three single models plateaued around 63%, which matched the earlier single-split results and confirmed the evaluation was representative.

---

## Ensemble Models

Ensemble models combine many models together to make a better collective decision. Two ensemble approaches were tested.

**Random Forest** builds hundreds of Decision Trees in parallel, each trained on a random subset of the data and features. The trees vote and the majority wins. The randomness forces the trees to be different from each other, which reduces overfitting.

**Gradient Boosting** builds trees sequentially. Each new tree focuses specifically on the mistakes made by the previous trees. It is slower but often more accurate than Random Forest.

### Ensemble Models — Cross-Validation Results

| Model               | Mean Accuracy | Std  |
|---------------------|:-------------:|:----:|
| Random Forest       | ~0.65+        | low  |
| Gradient Boosting   | ~0.65+        | low  |

The ensemble models outperformed the single models, demonstrating that combining weak learners produces a more robust result.

---

## Hyperparameter Tuning

The best performing model was tuned using `GridSearchCV`.

`GridSearchCV` tries every combination of hyperparameters specified in a grid, evaluates each combination with cross-validation internally, and returns the best configuration.

Parameters inside a pipeline are accessed with the `stepname__parametername` syntax:

```python
param_grid = {
    "classifier__n_estimators":      [100, 200, 300],
    "classifier__max_depth":         [None, 10, 20],
    "classifier__min_samples_split": [2, 5, 10],
}
```

With 3 × 3 × 3 = 27 combinations and 5-fold cross-validation, GridSearchCV ran 135 fits total.

`n_jobs=-1` was used to parallelize across all available CPU cores.

### Tuning Results

```text
Best Parameters : (from GridSearchCV output)
Best CV Score   : (from GridSearchCV output)
Final Test Accuracy : (from held-out test set)
```

---

## Early Experiments

### Unigrams vs Bigrams

The original TF-IDF representation used unigrams. Bigrams were introduced using `ngram_range=(1, 2)`.

This increased the feature space substantially:

```text
Unigrams → 1,888 features
Bigrams  → 10,008 features
```

However, accuracy only changed from 63.15% to 63.20%. Simply increasing the number of text features does not necessarily provide meaningful improvement. Feature quality matters more than feature count.

### Resume–Job Description Cosine Similarity

A cosine similarity feature was created to measure how similar each resume was to its corresponding job description using a shared TF-IDF vocabulary.

The correlation between similarity and the target was:

```text
-0.00945
```

The average similarity was almost identical between the two classes:

```text
Best Match = 0 → 0.227
Best Match = 1 → 0.225
```

Adding this feature did not improve performance. Simple lexical similarity between a resume and a job description was not enough to explain whether an application was a good match. Two texts can be semantically related while having low word-level overlap.

---

## Key Challenges and Lessons

### 1. Pipelines enforce correct data flow

Writing preprocessing manually across many cells works but is error-prone. A `Pipeline` with `ColumnTransformer` makes the workflow reproducible, readable, and safe from leakage by design.

### 2. Cross-validation gives a more honest evaluation

A single train/test split gives one number. Cross-validation gives a distribution. The mean and standard deviation together tell you how well the model performs and how stable that performance is.

### 3. Ensemble models outperform single models

Random Forest and Gradient Boosting consistently beat Logistic Regression and SVC. Combining many weak learners reduces variance and produces more robust predictions.

### 4. Hyperparameters matter — but search them systematically

Defaults are reasonable starting points. GridSearchCV finds better configurations without guessing, and cross-validation inside the search prevents overfitting to a single split.

### 5. Text similarity is not the same as semantic similarity

TF-IDF captures word occurrence and importance but does not understand that different words can represent the same concept. The cosine similarity experiment confirmed this limitation directly.

### 6. More features do not automatically mean a better model

Bigrams increased the feature space more than fivefold with almost no accuracy gain. Feature quality matters more than feature count.

### 7. A feature should be validated before relying on it

Cosine similarity looked like a reasonable job-matching feature but contained almost no predictive signal. Testing features against the target before including them is always worth doing.

---

## Final Results Summary

| Model                              | Accuracy |
|------------------------------------|:--------:|
| Majority class baseline            | 51.5%    |
| Logistic Regression (single split) | 63.15%   |
| Linear SVC                         | 63.05%   |
| Naive Bayes                        | 54.90%   |
| Logistic Regression + Bigrams      | 63.20%   |
| Logistic Regression + Similarity   | 63.15%   |
| Random Forest (CV)                 | ~65%+    |
| Gradient Boosting (CV)             | ~65%+    |
| Best Model after GridSearchCV      | (final)  |

---

## Final Takeaway

The most important outcome of this project was not achieving the highest possible accuracy.

It was building a complete, production-style NLP classification workflow — from raw text to tuned ensemble model — and experimentally identifying where classical approaches hit their ceiling.

```text
Manual preprocessing  → replaced by Pipeline + ColumnTransformer
Single split eval     → replaced by 5-fold cross-validation
Single models         → outperformed by ensemble models
Default parameters    → improved by systematic GridSearchCV tuning
TF-IDF features       → shown to plateau regardless of tuning
```

To move beyond this performance, the next logical step would be richer semantic representations such as word embeddings or transformer-based models — which appear in later projects in this series.

This project established a strong classical NLP baseline, introduced the full scikit-learn pipeline workflow, and provided a clear, experiment-backed reason to move toward semantic NLP techniques.
