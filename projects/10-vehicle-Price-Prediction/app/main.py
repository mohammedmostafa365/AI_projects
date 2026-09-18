from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

model = joblib.load("model.pkl")

class CarFeatures(BaseModel):
    condition: int
    drive: int
    fuel: int
    odometer: float
    paint_color: int
    title_status: int
    transmission: int
    type: int
    cylinders: int
    year: int
    make: int

@app.get("/")
def home():
    return {"message": "Car Price Predictor API"}

@app.post("/predict")
def predict(car: CarFeatures):
    features = np.array([[car.condition, car.drive, car.fuel, car.odometer,
                          car.paint_color, car.title_status, car.transmission,
                          car.type, car.cylinders, car.year, car.make]])
    prediction = model.predict(features)
    return {"predicted_price": round(prediction[0], 2)}