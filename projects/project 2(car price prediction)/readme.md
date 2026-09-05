# Day 2 — Used Car Fair Price Engine

## Overview

A regression model that predicts the fair market price of a used car based on its features. The goal is to help buyers and sellers identify whether a listed price is reasonable.

## Dataset

- **Source:** Kaggle — Used Cars Dataset
- **Size:** ~4,009 rows, 12 columns (after outlier removal: ~3,600+ rows)
- **Target:** `price` (continuous, USD)

## Dataset Limitations

The dataset was not clean or ready to use. Several issues made this project harder than expected:

- **Price was stored as a string** — values like `$10,300` had to be stripped of `$` and `,` and converted to float before anything could be done with them
- **Mileage was also a string** — stored as `"51,000 mi."` and required the same cleaning
- **Engine information was packed into one raw text column** — horsepower, engine size, and cylinder count all had to be extracted using regex patterns from strings like `"300.0HP 3.7L V6 Cylinder Engine"`
- **fuel_type had a dash character (`–`) used as a placeholder** — this is not a standard null value and had to be manually replaced before filling
- **clean_title only had one value (`Yes`)** — NaN meant "No", which had to be inferred and filled manually
- **Extreme price outliers existed in the data** — a small number of listings had prices in the hundreds of thousands or more, which are likely exotic or erroneous entries that don't belong in a general used car model

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

## Challenges & What Went Wrong

This project had a difficult debugging process. The first results were very bad and it took several iterations to understand why.

### First attempt results (before fixes)
| Model | R² |
|---|---|
| Random Forest | 0.13 |
| Linear Regression | 0.09 |
| Decision Tree | 0.06 |

These results looked like the model had learned almost nothing. The investigation found multiple stacked bugs:

**Bug 1 — `brand` was completely missing from the feature matrix.**
The encoding step one-hot encoded `model` (1,634 unique values) which exploded the feature space, but `brand` was silently dropped at some point and never made it into the model at all. Brand is one of the strongest price signals in any used car dataset.

**Bug 2 — `clean_title` was silently broken.**
NaN values were filled with the string `"no"` (lowercase), but the mapping dictionary used `"No"` (capital N). The mismatch meant all the filled values became NaN again after the map, making the column useless.

**Bug 3 — `price` was cleaned at the wrong point.**
The `$` and `,` stripping happened mid-notebook in some cells, after the train/test split in others. With out-of-order cell execution in Colab this caused inconsistency in what `y_train` and `y_test` actually contained.

**Bug 4 — Extreme price outliers were destroying RMSE.**
Even after fixing the encoding bugs, RMSE stayed around $133,000. The cause was a small number of listings with prices of $500,000 or more. RMSE squares errors, so a single car predicted at $30,000 but listed at $1,000,000 adds an enormous penalty. Removing these outlier rows before the split dropped RMSE from $133,000 to $9,400.

## Final Results

| Model | MAE | RMSE | R² |
|---|---|---|---|
| **Random Forest** | $6,317 | $9,372 | **0.7996** |
| Gradient Boosting | $6,369 | $9,477 | 0.7951 |
| Ridge | $7,995 | $12,544 | 0.6410 |
| Linear Regression | $8,086 | $12,942 | 0.6178 |
| Decision Tree | $8,892 | $13,559 | 0.5806 |

## Why the Results Are Still Not Perfect

Even after all fixes, an R² of ~0.80 means the model still cannot explain 20% of price variation. This is largely a data quality issue, not a model issue:

- The dataset is relatively small (~4,000 rows) for the number of unique car models (1,898)
- Color columns (`ext_col`, `int_col`) have hundreds of unique values with very sparse representation
- The `transmission` column had 62 unique values, many of which are near-identical descriptions of the same type
- A better dataset with more rows, consistent formatting, and standardized categories would likely push R² above 0.90

## Key Learnings

- **Outlier removal on the target was the single biggest improvement** — extreme price listings inflated RMSE from ~$133,000 to ~$9,400 after removal
- **Scaling does not affect tree-based models** — Random Forest and Gradient Boosting produce identical results with or without scaling; it only helps linear models
- **Frequency encoding is better than one-hot for high-cardinality columns** — encoding `brand` and `model` as frequency values avoids feature explosion while preserving signal
- **Silent bugs are the hardest to find** — the `clean_title` case study is a good example: the code ran without errors but the column was completely wrong
- **Cell execution order in Colab is dangerous** — running cells out of order caused bugs that were invisible until the full notebook was restarted and run top to bottom

## Files

- `day02_used_car_price_engine.ipynb` — main notebook
