from fastapi import FastAPI
from pydantic import BaseModel
import os
import joblib
import numpy as np

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model_path = os.path.join(BASE_DIR, "models", "fraud_model.pkl")
features_path = os.path.join(BASE_DIR, "models", "feature_columns.pkl")

model = joblib.load(model_path)
feature_columns = joblib.load(features_path)


app = FastAPI()

class Transaction(BaseModel):
    features: list

@app.get("/")
def home():
    return {"message": "Fraud Detection API Running"}

@app.post("/predict")
def predict(transaction: Transaction):
    data = np.array(transaction.features).reshape(1, -1)
    
    prediction = model.predict(data)[0]
    probability = model.predict_proba(data)[0][1]

    return {
        "fraud_prediction": int(prediction),
        "fraud_probability": float(probability)
    }
