from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import os

# 1. FastAPI Application එක ආරම්භ කිරීම
app = FastAPI(title="Liver Cancer Recurrence Prediction API", version="1.0")

# 2. Model එක සහ Scaler එක Load කරගැනීම
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'best_xgb_classifier.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'models', 'feature_scaler.pkl')

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
except Exception as e:
    print(f"Error loading models: {e}")

# 3. App එකෙන් එන දත්ත වල හැඩය (Input Schema) Pydantic හරහා සැකසීම
class PatientData(BaseModel):
    tumor_size_cm: float
    tumor_number: int
    vascular_invasion_imaging: int
    afp_ngml: float
    cirrhosis_present: int
    image_feature_vector_norm: float
    text_embedding_risk_score: float
    multimodal_feature_vector_norm: float

# 4. Prediction එක කරන POST Endpoint එක හැදීම
@app.post("/predict")
def predict_recurrence(data: PatientData):
    try:
        # Pydantic model එක Dictionary එකක් කරලා ඊටපස්සේ Pandas DataFrame එකක් බවට පත් කිරීම
        input_data = pd.DataFrame([data.dict()])
        
        # දත්ත Scaling කිරීම (Model එක Train කරපු විදිහටම)
        scaled_data = scaler.transform(input_data)
        
        # පුරෝකථනය (Prediction) සහ සම්භාවිතාව (Probability) ගැනීම
        prediction = model.predict(scaled_data)[0]
        probability = model.predict_proba(scaled_data)[0][1] * 100
        
        # ප්‍රතිඵලය App එකට යැවීම (JSON format එකෙන්)
        return {
            "prediction": int(prediction),
            "probability_percentage": round(float(probability), 2),
            "risk_level": "High Risk" if prediction == 1 else "Low Risk",
            "message": "මෙම රෝගියාට අක්මා පිළිකාව නැවත ඇතිවීමේ දැඩි අවදානමක් ඇත." if prediction == 1 else "අවදානම අඩු මට්ටමක පවතී."
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# App එක වැඩද කියලා බලන්න Root endpoint එකක්
@app.get("/")
def read_root():
    return {"message": "Welcome to the Liver Cancer Prediction API!"}