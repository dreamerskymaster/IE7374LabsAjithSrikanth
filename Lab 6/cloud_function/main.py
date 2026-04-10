"""Google Cloud Function (Gen 2) for CNC defect prediction.
Entry point: handle_request
Loads model and scaler from GCS at cold start, returns predictions as JSON,
logs predictions to BigQuery asynchronously."""

import os
import json
import uuid
import joblib
import tempfile
import pandas as pd
import functions_framework
from datetime import datetime

from utils import (
    engineer_features_from_raw,
    ALL_FEATURES,
    SENSOR_FEATURES,
    QUALITY_MAP,
)

# Global model cache (loaded once at cold start)
_model = None
_model_meta = None
_scaler = None
GCS_BUCKET = os.environ.get("GCS_BUCKET", "lab6-dvc-ajith")
BQ_DATASET = os.environ.get("BQ_DATASET", "cnc_warehouse")
PROJECT_ID = os.environ.get("PROJECT_ID", "ajithmlopsie7374")


def load_model():
    """Download model and scaler from GCS and cache in global scope."""
    global _model, _model_meta, _scaler
    if _model is not None and _scaler is not None:
        return

    from google.cloud import storage
    client = storage.Client()
    bucket = client.bucket(GCS_BUCKET)

    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
        bucket.blob("models/champion_model.pkl").download_to_filename(tmp.name)
        _model = joblib.load(tmp.name)

    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
        bucket.blob("models/feature_scaler.pkl").download_to_filename(tmp.name)
        _scaler = joblib.load(tmp.name)

    try:
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            bucket.blob("models/model_metadata.json").download_to_filename(tmp.name)
            with open(tmp.name) as f:
                _model_meta = json.load(f)
    except Exception:
        _model_meta = {"model_type": "RandomForest", "pipeline_version": "lab6-v1.0"}


def log_to_bigquery(predictions):
    """Best-effort async log of predictions to BigQuery."""
    try:
        from google.cloud import bigquery
        client = bigquery.Client(project=PROJECT_ID)
        table_id = f"{PROJECT_ID}.{BQ_DATASET}.prediction_log"
        rows_df = pd.DataFrame(predictions)
        rows_df["timestamp"] = pd.to_datetime(rows_df["timestamp"])
        # No .result() here to make it async/non-blocking
        client.load_table_from_dataframe(rows_df, table_id)
    except Exception as e:
        print(f"BigQuery Logging Error: {str(e)}")


def make_response(body, status=200):
    """Create a JSON response with CORS headers."""
    import flask
    resp = flask.make_response(json.dumps(body), status)
    resp.headers["Content-Type"] = "application/json"
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return resp


@functions_framework.http
def handle_request(request):
    """Main entry point for Cloud Function."""
    if request.method == "OPTIONS":
        return make_response({}, 204)

    action = "predict"
    body = {}

    if request.method == "GET":
        action = request.args.get("action", "health")
    elif request.method == "POST":
        if request.is_json:
            body = request.get_json(silent=True) or {}
            action = body.get("action", "predict")
        else:
            return make_response({"error": "Request must be JSON"}, 400)
    else:
        return make_response({"error": "Method not allowed"}, 405)

    if action == "health":
        return make_response({"status": "healthy", "timestamp": datetime.now().isoformat()})

    if action == "model-info":
        load_model()
        return make_response({"model": _model_meta})

    if action in ("predict", "batch-predict"):
        load_model()
        readings = body.get("readings", [])
        if not readings:
            return make_response(
                {"error": 'No readings provided. Send {"action": "predict", "readings": [{...}]}'},
                400,
            )

        try:
            df = pd.DataFrame(readings)
            missing = [c for c in SENSOR_FEATURES if c not in df.columns]
            if missing:
                return make_response({"error": f"Missing sensor fields: {missing}"}, 400)

            df_eng = engineer_features_from_raw(df)
            X = _scaler.transform(df_eng[ALL_FEATURES])

            preds = _model.predict(X)
            probas = _model.predict_proba(X)

            results = []
            log_rows = []
            for i, (pred, proba) in enumerate(zip(preds, probas)):
                pred_id = f"PRED-{str(uuid.uuid4())[:8]}"
                result = {
                    "prediction_id": pred_id,
                    "predicted_class": QUALITY_MAP[int(pred)],
                    "confidence": round(float(max(proba)), 4),
                    "probabilities": {QUALITY_MAP[j]: round(float(p), 4) for j, p in enumerate(proba)},
                }
                results.append(result)
                log_rows.append({
                    "prediction_id": pred_id,
                    "timestamp": datetime.now().isoformat(),
                    **{k: float(readings[i].get(k, 0)) for k in SENSOR_FEATURES},
                    "predicted_class": QUALITY_MAP[int(pred)],
                    "confidence": round(float(max(proba)), 4),
                    "model_version": _model_meta.get("pipeline_version", "unknown"),
                })

            log_to_bigquery(log_rows)

            return make_response({"predictions": results, "count": len(results)})

        except Exception as e:
            return make_response({"error": str(e)}, 500)

    return make_response({"error": f"Unknown action: {action}"}, 400)
