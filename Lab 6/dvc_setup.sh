#!/bin/bash
set -e
BUCKET="${GCS_BUCKET:-lab6-dvc-ajith}"
PROJECT="${GCP_PROJECT:-ajithmlopsie7374}"
echo "=== DVC Setup for Lab 6 ==="
gsutil mb -l us-central1 -p ${PROJECT} gs://${BUCKET} 2>/dev/null || echo "Bucket already exists"
dvc init --no-scm 2>/dev/null || echo "DVC already initialized"
dvc remote add -d gcs-remote gs://${BUCKET}/dvc-store -f
echo "DVC configured with GCS remote: gs://${BUCKET}/dvc-store"
