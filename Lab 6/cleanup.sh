#!/bin/bash
set -e
echo "=== Cleaning Lab 6 GCP Resources ==="
gcloud functions delete cnc-defect-predictor --region=us-central1 --gen2 --quiet 2>/dev/null || true
bq rm -r -f ajithmlopsie7374:cnc_warehouse 2>/dev/null || true
gsutil -m rm -r gs://lab6-dvc-ajith 2>/dev/null || true
echo "=== All Lab 6 resources cleaned up ==="
