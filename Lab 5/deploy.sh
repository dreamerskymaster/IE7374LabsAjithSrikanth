#!/bin/bash
set -e

PROJECT_ID="ajithmlopsie7374"
REGION="us-central1"
BUCKET_NAME="${PROJECT_ID}-mlops-artifacts"
REPO_NAME="mlops-repo"
IMAGE_NAME="cnc-prediction-service"
IMAGE_TAG="latest"
SERVICE_NAME="cnc-prediction-api"

echo "=== Deploying Lab 5 to GCP ==="
echo "Project: $PROJECT_ID"
echo "Region: $REGION"

# 1. Set Project
gcloud config set project $PROJECT_ID

# 2. Setup GCS Bucket
echo "Checking GCS Bucket: gs://$BUCKET_NAME..."
if ! gsutil ls "gs://$BUCKET_NAME" >/dev/null 2>&1; then
    echo "Creating bucket gs://$BUCKET_NAME..."
    gsutil mb -l $REGION "gs://$BUCKET_NAME"
else
    echo "Bucket gs://$BUCKET_NAME already exists."
fi

# 3. Setup Artifact Registry
echo "Checking Artifact Registry Repository: $REPO_NAME..."
if ! gcloud artifacts repositories describe $REPO_NAME --location=$REGION >/dev/null 2>&1; then
    echo "Creating repository $REPO_NAME..."
    gcloud artifacts repositories create $REPO_NAME \
        --repository-format=docker \
        --location=$REGION \
        --description="Docker repository for MLOps Lab 5"
else
    echo "Repository $REPO_NAME already exists."
fi

# Configure Docker to authenticate with Artifact Registry
gcloud auth configure-docker ${REGION}-docker.pkg.dev --quiet

# 4. Build and Push Docker Image
IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${IMAGE_NAME}:${IMAGE_TAG}"
echo "Building serving image for linux/amd64..."
docker build --platform linux/amd64 -f Dockerfile.serve -t $IMAGE_URI .

echo "Pushing image to Artifact Registry..."
docker push $IMAGE_URI

# 5. Deploy to Cloud Run
echo "Deploying to Cloud Run..."
gcloud run deploy $SERVICE_NAME \
    --image $IMAGE_URI \
    --region $REGION \
    --platform managed \
    --allow-unauthenticated \
    --memory 512Mi \
    --min-instances 0 \
    --max-instances 1 \
    --set-env-vars GCS_BUCKET=$BUCKET_NAME,CHAMPION_MODEL_FILE=champion_model.pkl

echo "=== Deployment Complete ==="
echo "Cloud Run Service URL:"
gcloud run services describe $SERVICE_NAME --region $REGION --format 'value(status.url)'
