from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import os

# 1. Initialize FastAPI Application
app = FastAPI(title="Liver Cancer Recurrence Prediction API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Load Model and Scaler
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'best_xgb_classifier.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'models', 'feature_scaler.pkl')

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
except Exception as e:
    print(f"Error loading models: {e}")

# 3. Define Input Schema using Pydantic
class PatientData(BaseModel):
    tumor_size_cm: float
    tumor_number: int
    vascular_invasion_imaging: int
    afp_ngml: float
    cirrhosis_present: int
    image_feature_vector_norm: float
    text_embedding_risk_score: float
    multimodal_feature_vector_norm: float

# 4. Create POST Endpoint for Prediction
@app.post("/predict")
def predict_recurrence(data: PatientData):
    try:
        # Convert Pydantic model to Dictionary, then to Pandas DataFrame
        input_data = pd.DataFrame([data.dict()])
        
        # Scale input data (matching the training configuration)
        scaled_data = scaler.transform(input_data)
        
        # Obtain Prediction and Probability
        prediction = model.predict(scaled_data)[0]
        probability = model.predict_proba(scaled_data)[0][1] * 100
        
        # Return result to the App (in JSON format)
        return {
            "prediction": int(prediction),
            "probability_percentage": round(float(probability), 2),
            "risk_level": "High Risk" if prediction == 1 else "Low Risk",
            "message": "This patient has a high risk of liver cancer recurrence." if prediction == 1 else "The risk is at a low level."
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Root endpoint to check if the App is running
@app.get("/")
def read_root():
    return {"message": "Welcome to the Liver Cancer Prediction API!"}