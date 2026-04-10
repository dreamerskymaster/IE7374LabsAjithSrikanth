#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

PROJECT="${GCP_PROJECT:-ajithmlopsie7374}"
REGION="${GCP_REGION:-us-central1}"
BUCKET="${GCS_BUCKET:-lab6-dvc-ajith}"
FUNCTION="${GCP_FUNCTION:-cnc-defect-predictor}"

echo "=== Deploying Cloud Function: ${FUNCTION} ==="

# Keep Cloud Function utils in sync with pipeline (single source of truth)
echo "[0/3] Syncing utils.py into cloud_function/..."
cp "${SCRIPT_DIR}/src/utils.py" "${SCRIPT_DIR}/cloud_function/utils.py"

# Upload model artifacts to GCS
echo "[1/3] Uploading model to GCS..."
gsutil cp models/champion_model.pkl gs://${BUCKET}/models/champion_model.pkl
gsutil cp models/model_metadata.json gs://${BUCKET}/models/model_metadata.json
gsutil cp models/feature_scaler.pkl gs://${BUCKET}/models/feature_scaler.pkl

# Deploy Gen 2 Cloud Function
echo "[2/3] Deploying function..."
gcloud functions deploy ${FUNCTION} \
  --gen2 \
  --region=${REGION} \
  --runtime=python311 \
  --source=cloud_function/ \
  --entry-point=handle_request \
  --trigger-http \
  --allow-unauthenticated \
  --memory=512MB \
  --timeout=60s \
  --min-instances=0 \
  --max-instances=2 \
  --set-env-vars="GCS_BUCKET=${BUCKET},BQ_DATASET=cnc_warehouse,PROJECT_ID=${PROJECT}" \
  --quiet

# Print URL
echo "[3/3] Getting URL..."
URL=$(gcloud functions describe ${FUNCTION} --region=${REGION} --gen2 --format='value(serviceConfig.uri)')
echo ""
echo "=== Deployed Successfully ==="
echo "URL: ${URL}"
echo ""
echo "Test health:   curl \"${URL}?action=health\""
echo "Test info:     curl \"${URL}?action=model-info\""
echo "Test predict:  curl -X POST ${URL} -H 'Content-Type: application/json' \\"
echo "  -d '{\"action\":\"predict\",\"readings\":[{\"spindle_speed\":2500,\"feed_rate\":200,\"depth_of_cut\":2.0,\"vibration\":6.5,\"temperature\":290,\"tool_wear\":0.45}]}'"
