import os
import json
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google.cloud import storage
import uvicorn
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PredictionService")

app = FastAPI(title="CNC Defect Prediction API (Lab 5)", version="1.0")

MODEL_DIR = "/app/model"
GCS_BUCKET = os.environ.get("GCS_BUCKET")
CHAMPION_MODEL_FILE = os.environ.get("CHAMPION_MODEL_FILE", "champion_model.pkl")

# We will load the model globally
champion_model = None

class PredictionRequest(BaseModel):
    features: dict

class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    model_used: str

def download_model_from_gcs():
    """Downloads the champion model from GCS if available."""
    if not GCS_BUCKET:
        logger.warning("GCS_BUCKET not set, skipping GCS download.")
        return False
    
    try:
        client = storage.Client()
        bucket = client.bucket(GCS_BUCKET)
        
        # In a real scenario, we'd read the model_comparison_summary.json to find the champion
        # For simplicity, we assume deploy.sh names the best model "champion_model.pkl" in GCS
        # or we download a specific blob.
        blob = bucket.blob(f"models/{CHAMPION_MODEL_FILE}")
        
        os.makedirs(MODEL_DIR, exist_ok=True)
        local_path = os.path.join(MODEL_DIR, CHAMPION_MODEL_FILE)
        
        if blob.exists():
            blob.download_to_filename(local_path)
            logger.info(f"Downloaded model from GCS to {local_path}")
            return local_path
        else:
            logger.warning(f"Blob models/{CHAMPION_MODEL_FILE} not found in GCS.")
            return False
    except Exception as e:
        logger.error(f"Error downloading from GCS: {e}")
        return False

def load_local_model():
    """Tries to load model from local path first fallback."""
    local_path = os.path.join(MODEL_DIR, CHAMPION_MODEL_FILE)
    if os.path.exists(local_path):
        return local_path
    
    # Try looking in results if mounted
    results_path = os.path.join("/opt/airflow/results", CHAMPION_MODEL_FILE)
    if os.path.exists(results_path):
        return results_path
        
    return None

@app.on_event("startup")
def startup_event():
    global champion_model
    logger.info("Starting up Prediction Service...")
    
    model_path = download_model_from_gcs()
    if not model_path:
        model_path = load_local_model()
        
    if model_path:
        try:
            champion_model = joblib.load(model_path)
            logger.info(f"Successfully loaded model from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {e}")
    else:
        logger.warning("No model found on startup. API will return 503 for predictions until model is available.")

@app.get("/health")
def health_check():
    status = "healthy" if champion_model is not None else "degraded (no model loaded)"
    return {"status": status, "model_loaded": champion_model is not None}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if champion_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Service unavailable.")
        
    try:
        df = pd.DataFrame([request.features])
        prediction = champion_model.predict(df)[0]
        
        # Check if model supports predict_proba
        if hasattr(champion_model, "predict_proba"):
            probability = champion_model.predict_proba(df)[0][1]
        else:
            probability = float(prediction)
            
        return PredictionResponse(
            prediction=int(prediction),
            probability=float(probability),
            model_used="Champion Model"
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
