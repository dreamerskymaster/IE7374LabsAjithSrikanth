from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np
import os

# --- API Configuration & Initialization ---
# Author: Ajith Srikanth
# Project: IE7374 MLOps - Lab 2 (FastAPI Backend)

app = FastAPI(
    title="Iris Species Prediction API",
    description="A robust FastAPI backend serving a Random Forest model for botanical classification.",
    version="1.0.0"
)

# Define the expected input schema for our prediction endpoint
class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Global variable to hold our trained model
MODEL_PATH = "Lab 2/backend/model/iris_model.pkl"
model = None

@app.on_event("startup")
def load_model():
    """
    Service Lifecycle: Startup
    Loads the serialized model from disk into memory for rapid inference.
    """
    global model
    if os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH, 'rb') as f:
                model = pickle.load(f)
            print(f"Successfully deployed model from {MODEL_PATH}")
        except Exception as e:
            print(f"Critical Error: Failed to load model. {e}")
    else:
        print(f"Warning: Model file not found at {MODEL_PATH}. Please run training script first.")

# --- API Endpoints ---

@app.get("/")
def health_check():
    """
    Endpoint: Health Check
    Returns the operational status of the API and model readiness.
    """
    return {
        "status": "online",
        "message": "Ajith's Iris Prediction API is ready to serve!",
        "model_loaded": model is not None
    }

@app.post("/predict")
def predict_species(features: IrisFeatures):
    """
    Endpoint: species Prediction
    
    Processes morphological features of an Iris flower and returns 
     the predicted species classification.
    
    Args:
        features (IrisFeatures): JSON payload containing length/width of sepals and petals.
        
    Returns:
        dict: The classification result (Setosa, Versicolor, or Virginica).
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Prediction service unavailable: Model not loaded.")
    
    try:
        # Transform input features into the format expected by Scikit-Learn (2D array)
        data = np.array([[
            features.sepal_length, 
            features.sepal_width, 
            features.petal_length, 
            features.petal_width
        ]])
        
        # Perform inference
        prediction = model.predict(data)
        
        # Map numeric prediction back to botanical labels
        species_map = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}
        result = species_map.get(int(prediction[0]), "Unknown species")
        
        return {
            "prediction": result,
            "input_features": features.dict()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference Engine Error: {str(e)}")
