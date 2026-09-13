
# Day 1 — Food Waste Predictor

## Goal

Predict the amount of food wasted (in kg) per month based on operational data from a food service environment.

This project focuses on building a complete data analysis and machine learning workflow from scratch — from raw messy data to a trained and evaluated regression model.

---

## Concepts Covered

- **Exploratory Data Analysis (EDA)** — understanding the data, spotting patterns, identifying problems
- **Data Cleaning** — handling missing values, fixing data types, parsing dates
- **Outlier Detection & Handling** — using IQR to detect and treat outliers differently in train vs test data
- **Feature Engineering** — extracting year, month, and day from a date column
- **Encoding** — converting categorical features to numerical using Label Encoding and One-Hot Encoding
- **Scaling** — standardizing numerical features using StandardScaler
- **Feature Selection** — using a correlation matrix to identify the most relevant features for the target
- **Regression Modeling** — training and comparing multiple models: Linear Regression, Random Forest, Gradient Boosting, and KNN
- **Model Evaluation** — using R² and Mean Squared Error (MSE) on a held-out validation set
- **Ensemble Learning** — combining models using a Voting Regressor and analyzing the result

---

## Dataset

**Source:** Kaggle — Messy Food Waste Prediction Dataset
**Link:** https://www.kaggle.com/competitions/messy-food-waste-prediction-dataset/data
**Size:** ~900 rows

---

## Key Takeaway

Random Forest outperformed all other models on the validation set. The ensemble did not improve results because averaging weak models (Linear Regression, KNN) alongside strong ones pulled the overall performance down.
