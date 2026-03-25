import os
import json
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
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

@app.get("/", response_class=HTMLResponse)
def read_root():
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>CNC Defect Prediction</title>
        <style>
            :root {
                --bg: #0f172a; --panel: #1e293b; --text: #f8fafc;
                --accent: #3b82f6; --accent-hover: #2563eb;
                --success: #10b981; --warn: #f59e0b; --danger: #ef4444;
            }
            body {
                font-family: 'Inter', system-ui, sans-serif;
                background-color: var(--bg); color: var(--text);
                margin: 0; padding: 2rem; display: flex; justify-content: center;
            }
            .container {
                background: var(--panel); max-width: 600px; width: 100%;
                border-radius: 12px; padding: 2rem; box-shadow: 0 4px 6px rgba(0,0,0,0.3);
            }
            h1 { margin-top: 0; font-size: 1.5rem; text-align: center; color: var(--accent); }
            p.subtitle { text-align: center; color: #94a3b8; font-size: 0.9rem; margin-bottom: 2rem; }
            .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }
            .form-group { display: flex; flex-direction: column; }
            label { font-size: 0.8rem; margin-bottom: 0.25rem; color: #cbd5e1; }
            input {
                background: #334155; border: 1px solid #475569; color: white;
                padding: 0.5rem; border-radius: 6px; font-size: 0.9rem; outline: none;
            }
            input:focus { border-color: var(--accent); }
            .full-width { grid-column: 1 / -1; display: flex; gap: 1rem; margin-top: 1rem; }
            button {
                flex: 1; padding: 0.75rem; border: none; border-radius: 6px;
                font-weight: 600; cursor: pointer; transition: background 0.2s;
            }
            .btn-primary { background: var(--accent); color: white; }
            .btn-primary:hover { background: var(--accent-hover); }
            .btn-secondary { background: #475569; color: white; }
            .btn-secondary:hover { background: #64748b; }
            #result {
                margin-top: 2rem; padding: 1rem; border-radius: 8px;
                text-align: center; display: none; font-weight: bold; font-size: 1.2rem;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>CNC Defect Oracle</h1>
            <p class="subtitle">Live Predictions from the Cloud Run Champion Model</p>
            
            <form id="predict-form" class="grid">
                <div class="form-group"><label>Spindle Speed (RPM)</label><input type="number" id="spindle" step="0.1" value="2100.5" required></div>
                <div class="form-group"><label>Feed Rate (mm/min)</label><input type="number" id="feed" step="0.1" value="150.2" required></div>
                <div class="form-group"><label>Depth of Cut (mm)</label><input type="number" id="depth" step="0.1" value="2.1" required></div>
                <div class="form-group"><label>Vibration (mm/s²)</label><input type="number" id="vib" step="0.1" value="1.5" required></div>
                <div class="form-group"><label>Temperature (°C)</label><input type="number" id="temp" step="0.1" value="180.0" required></div>
                <div class="form-group"><label>Tool Wear Index</label><input type="number" id="wear" step="0.01" value="0.2" required></div>
                
                <div class="full-width">
                    <button type="button" class="btn-secondary" onclick="randomize()">🎲 Randomize Data</button>
                    <button type="submit" class="btn-primary">🚀 Run Prediction</button>
                </div>
            </form>
            
            <div id="result"></div>
        </div>

        <script>
            function randomize() {
                document.getElementById('spindle').value = (800 + Math.random() * 2400).toFixed(1);
                document.getElementById('feed').value = (50 + Math.random() * 350).toFixed(1);
                document.getElementById('depth').value = (0.5 + Math.random() * 3.5).toFixed(1);
                document.getElementById('vib').value = (0.1 + Math.random() * 7.9).toFixed(1);
                document.getElementById('temp').value = (150 + Math.random() * 200).toFixed(1);
                document.getElementById('wear').value = (Math.random() * 1).toFixed(2);
            }

            document.getElementById('predict-form').addEventListener('submit', async (e) => {
                e.preventDefault();
                const btn = document.querySelector('.btn-primary');
                btn.innerText = '⏳ Processing...';
                btn.disabled = true;
                
                // Read and calculate engineered features automatically for the user
                const spd = parseFloat(document.getElementById('spindle').value);
                const feed = parseFloat(document.getElementById('feed').value);
                const depth = parseFloat(document.getElementById('depth').value);
                const vib = parseFloat(document.getElementById('vib').value);
                const temp = parseFloat(document.getElementById('temp').value);
                const wear = parseFloat(document.getElementById('wear').value);
                
                const features = {
                    "spindle_speed": spd,
                    "feed_rate": feed,
                    "depth_of_cut": depth,
                    "vibration": vib,
                    "temperature": temp,
                    "tool_wear": wear,
                    "speed_feed_ratio": spd / feed,
                    "vibration_temp_interaction": vib * temp,
                    "tool_wear_severity": wear * depth,
                    "cutting_energy_proxy": spd * feed * depth,
                    "thermal_stress_index": temp * wear
                };

                try {
                    const res = await fetch('/predict', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ features })
                    });
                    
                    const data = await res.json();
                    const resultDiv = document.getElementById('result');
                    resultDiv.style.display = 'block';
                    
                    if (!res.ok) throw new Error(data.detail || 'Prediction failed');
                    
                    let cls = '', text = '';
                    if (data.prediction === 0) { cls = 'var(--success)'; text = 'Status: GOOD'; }
                    else if (data.prediction === 1) { cls = 'var(--warn)'; text = 'Status: MINOR DEFECT'; }
                    else { cls = 'var(--danger)'; text = 'Status: MAJOR DEFECT'; }
                    
                    const prob = (data.probability * 100).toFixed(1);
                    resultDiv.style.backgroundColor = cls;
                    resultDiv.style.color = '#fff';
                    resultDiv.innerHTML = `${text}<br><span style="font-size:0.9rem;font-weight:normal">Defect Probability: ${prob}%</span>`;
                } catch (err) {
                    alert("Error: " + err.message);
                } finally {
                    btn.innerText = '🚀 Run Prediction';
                    btn.disabled = false;
                }
            });
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

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
