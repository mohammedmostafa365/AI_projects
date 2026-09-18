# Day 10 — AI Decision Platform: Used Car Price Predictor

## Project Overview

An end-to-end machine learning application that predicts used car prices based on
vehicle features. This project covers the full deployment pipeline — from data
cleaning and model training to serving predictions via a REST API and a
interactive web interface.

## Objectives

- Build a complete ML pipeline from raw data to deployed model
- Learn how to serve a model using FastAPI
- Build a frontend UI using Streamlit
- Understand how the backend and frontend communicate via HTTP

## Project Structure

project 10 Vehicle Price Prediction/
├── data/
│ └── vehicles.csv
├── app/
│ ├── model.pkl
│ ├── main.py
│ └── streamlit_app.py
├── DAY10-Vehicle-Price-Prediction.ipynb
└── README.md


## Dataset

- **Source:** Kaggle — Craigslist Used Cars Dataset (austinreese)
- **Size:** ~2,600 rows, 21 columns
- **Target:** `price`

## Workflow

### 1. Data Cleaning
- Dropped irrelevant columns: `location`, `lat`, `long`, `Listed_date`,
  `Listed_time`, `year make model`, `re_model`, `size`
- Filled categorical nulls with `"unknown"`
- Filled odometer nulls with median
- Dropped rows with very few nulls in critical columns

### 2. Outlier Removal
- Filtered price to $500 – $150,000
- Filtered odometer to 1,000 – 500,000 miles

### 3. Feature Engineering
- Label encoded all categorical columns
- Final features: `condition`, `drive`, `fuel`, `odometer`, `paint_color`,
  `title_status`, `transmission`, `type`, `cylinders`, `year`, `make`

### 4. Model Training
- **Model:** Random Forest Regressor
- `n_estimators=200`, `max_depth=20`, `random_state=42`
- **MAE:** $5,582
- **R² Score:** 0.72

### 5. Model Saving
- Saved using `joblib` as `model.pkl`

## API — FastAPI

- Framework: FastAPI + Uvicorn
- Endpoint: `POST /predict`
- Input: JSON with 11 car features
- Output: `{ "predicted_price": float }`

**To run:**
```bash
cd app
uvicorn main:app --reload
```

API docs available at: `http://localhost:8000/docs`

## Frontend — Streamlit

- User selects car features via sliders and dropdowns
- Sends request to FastAPI `/predict` endpoint
- Displays predicted price

**To run (in a separate terminal):**
```bash
cd app
streamlit run streamlit_app.py
```

## What I Learned

- How to save and load a trained model using `joblib`
- How to build a REST API with FastAPI and expose ML model predictions
- How to build an interactive frontend with Streamlit
- How FastAPI and Streamlit communicate over HTTP
- The standard pattern for ML deployment: separate model, backend, and frontend

## Known Limitations

- Categorical features are label encoded — encoding is not saved, so new
  categories at inference time may not map correctly
- Several features (paint color, make, type) are hardcoded in the UI due to
  this limitation
- Small dataset (2,466 rows after cleaning) limits model performance

## Day Summary

This was the first time building a fully deployed ML application from scratch —
no tutorials, no vibe coding. FastAPI and Streamlit were both new. The most
important lesson is understanding the architecture: the model, the API, and the
UI are three separate pieces that communicate with each other. This pattern
applies to any ML deployment in the real world.
