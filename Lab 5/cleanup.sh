#!/bin/bash

PROJECT_ID="ajithmlopsie7374"
REGION="us-central1"
BUCKET_NAME="${PROJECT_ID}-mlops-artifacts"
REPO_NAME="mlops-repo"
SERVICE_NAME="cnc-prediction-api"

echo "=== Cleaning up Lab 5 GCP Resources ==="
echo "Project: $PROJECT_ID"

# 1. Delete Cloud Run Service
echo "Deleting Cloud Run service $SERVICE_NAME..."
gcloud run services delete $SERVICE_NAME --region $REGION --quiet || true

# 2. Delete Artifact Registry Repository
echo "Deleting Artifact Registry repository $REPO_NAME..."
gcloud artifacts repositories delete $REPO_NAME --location $REGION --quiet || true

# 3. Empty and Delete GCS Bucket
echo "Deleting GCS bucket gs://$BUCKET_NAME..."
gsutil -m rm -r "gs://$BUCKET_NAME" || true

echo "=== Cleanup Complete ==="
