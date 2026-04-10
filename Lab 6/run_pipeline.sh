#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

echo "============================================="
echo "  Lab 6: Full Pipeline Execution"
echo "  Cloud Functions + DVC + Data Warehouse"
echo "============================================="

echo ""
echo "[1/8] Setting up DVC with GCS remote..."
bash dvc_setup.sh

echo ""
echo "[2/8] Setting up BigQuery warehouse..."
bash setup_warehouse.sh

echo ""
echo "[3/8] Running DVC pipeline (generate > label > featurize > train)..."
dvc repro

echo ""
echo "[4/8] Versioning artifacts with DVC..."
dvc push
echo "Artifacts pushed to GCS remote"

echo ""
echo "[5/8] Tagging version in Git..."
if git rev-parse --git-dir > /dev/null 2>&1; then
  git add dvc.lock models/metrics.json .dvc/ dvc.yaml .dvcignore
  git add data/.gitignore models/.gitignore 2>/dev/null || true
  git commit -m "Lab 6 pipeline run $(date +%Y%m%d-%H%M%S)" || echo "Nothing new to commit"
  git tag -a "v1.0-lab6" -m "Lab 6: initial pipeline run" 2>/dev/null || echo "Tag v1.0-lab6 already exists"
else
  echo "  Skipped: not a Git repository. Run 'git init' in ${SCRIPT_DIR} to enable commits and tags."
fi

echo ""
echo "[6/8] Loading data into BigQuery warehouse..."
python -m src.warehouse_loader

echo ""
echo "[7/8] Running data quality checks..."
python -m src.data_quality

echo ""
echo "[8/8] Deploying Cloud Function..."
bash deploy_function.sh

echo ""
echo "============================================="
echo "  Pipeline Complete!"
echo "============================================="
echo "BigQuery Console: https://console.cloud.google.com/bigquery?project=ajithmlopsie7374"
echo "DVC Remote:       gs://lab6-dvc-ajith/dvc-store"
echo "DVC Metrics:      dvc metrics show"
echo "Restore v1.0:     git checkout v1.0-lab6 && dvc checkout"
