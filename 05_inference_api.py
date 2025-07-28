"""
05_inference_api.py

Create a FastAPI app to serve the student performance prediction model.
This API:
- Loads the trained model from models/best_model.pkl
- Defines a POST endpoint `/predict` that accepts student features as input
- Returns the predicted final grade G3 as JSON
"""

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import os

# Define input schema
class StudentData(BaseModel):
    G1: float
    G2: float
    studytime: float
    failures: int
    absences: int
    # Add more fields if your model uses more features

# Load model
MODEL_PATH = "models/best_model.pkl"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
model = joblib.load(MODEL_PATH)

# Create app
app = FastAPI()

@app.post("/predict")
def predict(data: StudentData):
    # Prepare input in the order expected by the model
    input_data = np.array([[data.G1, data.G2, data.studytime, data.failures, data.absences]])
    prediction = model.predict(input_data)
    return {"predicted_G3": round(float(prediction[0]), 2)}
