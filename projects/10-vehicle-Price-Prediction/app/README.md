
# Used Car Price Predictor — Local Setup

A machine learning app that predicts used car prices via a FastAPI backend and Streamlit frontend.

## Requirements

```bash
pip install fastapi uvicorn streamlit scikit-learn pandas joblib numpy
```

## How to Run

### 1. Start the API

Open a terminal inside the `app` folder and run:

```bash
uvicorn main:app --reload
```

API will be available at `http://localhost:8000`
Interactive docs at `http://localhost:8000/docs`

### 2. Start the Frontend

Open a second terminal inside the `app` folder and run:

```bash
streamlit run streamlit_app.py
```

Streamlit will open automatically in your browser.

## Notes

- Both terminals must be running at the same time
- The API must start before you use the Streamlit app
- `model.pkl` must be present in the `app` folder
