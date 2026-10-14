import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

MODEL_PATH = os.getenv("MODEL_PATH", "artifacts/model.joblib")

# Chargement du modèle
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    model = None

@app.get("/health")
def health():
    if model is None:
        raise HTTPException(status_code=500, detail="Modèle non chargé")
    return {"status": "ok"}

@app.post("/predict")
def predict(data: dict):
    if model is None:
        raise HTTPException(status_code=500, detail="Modèle non disponible")
    try:
        df = pd.DataFrame([data])
        df.columns = df.columns.str.strip().str.replace('-', '_')
        prediction = model.predict(df)
        return {"prediction": prediction[0]}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
