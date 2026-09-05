

# Advanced AI Practical Milestone 🚀

A 10-day hands-on challenge to move from theoretical knowledge to real implementation across the full Machine Learning and AI stack.

Each day is a standalone project targeting a specific area of the course — built from scratch, using real datasets, and following a complete ML workflow without relying on step-by-step tutorials.

---

## Goal

The objective is not to achieve the highest score.

The objective is to **understand what I am doing**, make independent decisions, debug real problems, and build complete ML and AI workflows — end to end.

---

## Project Progression

```
Data Analysis → Machine Learning → Advanced ML → Unsupervised Learning
→ Anomaly Detection → NLP → Neural Networks → Computer Vision
→ RNN / LSTM / GRU → Transformers → MLflow → Deployment
```

---

## Projects

### Day 1 — Food Waste Predictor
**Concepts:** Data Analysis · EDA · Data Cleaning · Visualization · Feature Engineering · Regression · Classification · Model Evaluation

Predict the amount of food wasted per month based on operational and environmental features. The project covers the full data pipeline from raw exploration to model comparison.

---

### Day 2 — Used Car Fair Price Engine
**Concepts:** Regression · Feature Engineering · Outlier Detection · Feature Selection · Scaling · Model Comparison · Hyperparameter Tuning

Estimate a fair market price for used cars. Focuses on handling messy real-world data, engineering meaningful features, and tuning models to improve prediction accuracy.

---

### Day 3 — Job Application Intelligence
**Concepts:** Classification · Preprocessing · Pipelines · Cross-Validation · Ensemble Models · Evaluation

Predict the outcome of job applications based on candidate and job features. Emphasizes building robust sklearn pipelines, ensemble strategies, and proper cross-validation.

---

### Day 4 — Customer Behavior Clustering
**Concepts:** Unsupervised Learning · K-Means · PCA · Dimensionality Reduction · Clustering Evaluation · Visualization

Segment customers into behavioral groups without labeled data. Covers the full unsupervised workflow — choosing K, reducing dimensions with PCA, and interpreting clusters meaningfully.

---

### Day 5 — Financial Transaction Anomaly Detector
**Concepts:** Anomaly Detection · IQR · Z-Score · Isolation Forest · LOF · Elliptic Envelope

Detect suspicious or fraudulent transactions in financial data using both statistical and machine learning approaches. Compares multiple anomaly detection strategies and their trade-offs.

---

### Day 6 — Arabic Review Intelligence
**Concepts:** NLP · Text Preprocessing · Tokenization · Stopwords · Stemming · Lemmatization · Bag of Words · TF-IDF · Word Embeddings · Text Classification

Classify Arabic customer reviews using NLP techniques. Covers the full Arabic text preprocessing pipeline and compares classical vectorization methods with word embeddings.

---

### Day 7 — Smart Traffic Vision
**Concepts:** Neural Networks · CNN · Image Preprocessing · Data Augmentation · Transfer Learning · PyTorch

Classify traffic scenes and road conditions from images. Builds and trains CNNs in PyTorch, applies data augmentation, and uses transfer learning to improve performance on limited data.

---

### Day 8 — Video Activity Detector
**Concepts:** CNN Feature Extraction · Sequence Processing · RNN · LSTM · GRU · BiLSTM · PyTorch

Recognize human activities from video sequences by combining CNN-based spatial feature extraction with recurrent architectures to model temporal patterns across frames.

---

### Day 9 — Arabic Document Intelligence
**Concepts:** Transformers · Hugging Face · Tokenization · Fine-Tuning · Transfer Learning · NLP Classification

Fine-tune a pre-trained Arabic transformer model (such as AraBERT) on a document classification task. Covers the Hugging Face ecosystem, tokenization, and transfer learning for Arabic NLP.

---

### Day 10 — AI Decision Platform
**Concepts:** End-to-End ML · Deep Learning · NLP · MLflow · Experiment Tracking · FastAPI · Streamlit · Model Serving · Deployment

A complete production-style AI platform that brings together models from multiple domains, tracks experiments with MLflow, serves predictions via FastAPI, and provides a Streamlit interface for interaction.

---

## Workflow (Per Project)

Every project follows this structure:

```
Problem Definition → Dataset → Data Understanding → EDA → Preprocessing
→ Baseline Model → Feature Engineering → Model Training → Evaluation
→ Error Analysis → Improvement → Documentation
```

---

## Stack

| Area | Tools |
|---|---|
| Data | pandas · numpy |
| Visualization | matplotlib · seaborn |
| Machine Learning | scikit-learn |
| Deep Learning | PyTorch |
| NLP | NLTK · Hugging Face Transformers |
| Experiment Tracking | MLflow |
| Deployment | FastAPI · Streamlit |

---

## Structure

```
/
├── day-01-food-waste-predictor/
├── day-02-used-car-price-engine/
├── day-03-job-application-intelligence/
├── day-04-customer-behavior-clustering/
├── day-05-financial-anomaly-detector/
├── day-06-arabic-review-intelligence/
├── day-07-smart-traffic-vision/
├── day-08-video-activity-detector/
├── day-09-arabic-document-intelligence/
└── day-10-ai-decision-platform/
```

---

*Built as part of an Advanced Machine Learning and AI course — 10 days, 10 projects, no tutorials.*
