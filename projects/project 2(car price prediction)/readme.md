
# Day 2 — Used Car Fair Price Engine

## Overview

A regression model that predicts the fair market price of a used car based on its features. The goal is to help buyers and sellers identify whether a listed price is reasonable.

## Dataset

- **Source:** Kaggle — Used Cars Dataset
- **Size:** ~4,009 rows, 12 columns (after outlier removal: ~3,600+ rows)
- **Target:** `price` (continuous, USD)

## Features Used

| Feature | Type | Notes |
|---|---|---|
| brand | categorical | frequency encoded |
| model | categorical | frequency encoded |
| model_year | numeric | |
| car_age | numeric | engineered: 2025 - model_year |
| milage | numeric | cleaned from string format |
| fuel_type | categorical | one-hot encoded |
| transmission | categorical | one-hot encoded |
| ext_col | categorical | frequency encoded |
| int_col | categorical | frequency encoded |
| accident | binary | 0 = none reported, 1 = accident reported |
| clean_title | binary | 0 = No, 1 = Yes |
| horsepower | numeric | extracted from engine string |
| engine_size | numeric | extracted from engine string |
| cylinders | numeric | extracted from engine string |

## Workflow

1. **Data Loading & EDA** — shape, nulls, duplicates, distributions
2. **Target Cleaning** — stripped `$` and `,` from price before split
3. **Outlier Removal** — removed rows where price was outside IQR bounds (before split)
4. **Train/Test Split** — 80/20, random_state=42
5. **Null Handling** — mode fill for fuel_type, accident; 'No' fill for clean_title
6. **Feature Engineering** — extracted horsepower, engine_size, cylinders from engine string; added car_age
7. **Encoding** — frequency encoding for high-cardinality columns; one-hot for low-cardinality
8. **Scaling** — StandardScaler on numeric columns (fit on train only)
9. **Model Training** — 5 models compared
10. **Evaluation** — MAE, RMSE, R²

## Results

| Model | MAE | RMSE | R² |
|---|---|---|---|
| **Random Forest** | $6,317 | $9,372 | **0.8996** |
| Gradient Boosting | $6,369 | $9,477 | 0.7951 |
| Ridge | $7,995 | $12,544 | 0.6410 |
| Linear Regression | $8,086 | $12,942 | 0.6178 |
| Decision Tree | $8,892 | $13,559 | 0.5806 |

## Key Learnings

- **Outlier removal on the target was the single biggest improvement** — a small number of extreme price listings ($500k+) inflated RMSE from ~$133,000 to ~$9,400 after removal
- **Scaling does not affect tree-based models** — Random Forest and Gradient Boosting produce identical results with or without scaling; it only helps linear models
- **Frequency encoding is better than one-hot for high-cardinality columns** — encoding `brand` and `model` as frequency values avoids feature explosion while preserving signal
- **Feature extraction from raw strings adds real value** — horsepower, engine size, and cylinders parsed from the engine column were meaningful predictors

## Files

- `day02_used_car_price_engine.ipynb` — main notebook
