import streamlit as st
import requests

st.title("Used Car Price Predictor")
st.write("Enter the car details to get a predicted price.")

year = st.slider("Year", 1990, 2024, 2015)
odometer = st.number_input("Odometer (miles)", min_value=1000, max_value=500000, value=90000)
condition = st.selectbox("Condition", options=[0, 1, 2, 3, 4, 5, 6], format_func=lambda x: ["excellent", "fair", "good", "like new", "new", "salvage", "unknown"][x])
transmission = st.selectbox("Transmission", options=[0, 1, 2], format_func=lambda x: ["automatic", "manual", "other"][x])
fuel = st.selectbox("Fuel", options=[0, 1, 2, 3, 4], format_func=lambda x: ["diesel", "electric", "gas", "hybrid", "other"][x])
drive = st.selectbox("Drive", options=[0, 1, 2, 3], format_func=lambda x: ["4wd", "fwd", "rwd", "unknown"][x])

if st.button("Predict Price"):
    payload = {
        "condition": condition,
        "drive": drive,
        "fuel": fuel,
        "odometer": odometer,
        "paint_color": 8,
        "title_status": 0,
        "transmission": transmission,
        "type": 9,
        "cylinders": 4,
        "year": year,
        "make": 80
    }

    response = requests.post("http://localhost:8000/predict", json=payload)
    result = response.json()
    st.success(f"Predicted Price: ${result['predicted_price']:,.2f}")