# Lab 5: MLOps Defect Pipeline on GCP (Airflow + MLflow + MLMD + TFDV)

![Python](https://img.shields.io/badge/Python-3.9-blue)
![Airflow](https://img.shields.io/badge/Apache_Airflow-2.7-017CEE)
![MLflow](https://img.shields.io/badge/MLflow-2.12-0194E2)
![MLMD](https://img.shields.io/badge/ML_Metadata-1.14-FF6F00)
![GCP](https://img.shields.io/badge/Google_Cloud-Serverless-4285F4)

## Overview

This project builds upon Lab 4 by extending the CNC manufacturing defect detection pipeline into a production-grade MLOps system on **Google Cloud Platform (GCP)**. Key additions include:

1. **TensorFlow Data Validation (TFDV)** for automated data profiling, schema generation, and anomaly detection.
2. **ML Metadata (MLMD)** for full artifact and execution lineage tracking across the pipeline.
3. **Advanced Hyperparameter Tuning** using `GridSearchCV` across robust parameter grids for all models.
4. **GCP Deployment** using Cloud Storage (GCS) for artifacts, Artifact Registry for Docker images, and Cloud Run for serverless model serving (staying under the free tier budget of $5).
5. **Interactive Lineage Dashboard** to visualize the complete lifecycle of ML artifacts and executions.

## Architecture

```mermaid
graph TD
    subgraph Local / Docker Compose
        A[Airflow DAG] -->|Orchestrates| B(Data Generation)
        B --> C(TFDV Validation)
        C --> D(Feature Engineering)
        D --> E{Parallel Training <br> RF, XGB, LGB, NN}
        E --> F(Evaluate & Compare)
        F --> G(Register Champion)
    end
    
    subgraph Metadata & Tracking
        B -.->|Logs Artifacts| MLMD[(ML Metadata SQLite)]
        C -.->|Logs Anomalies| MLMD
        D -.->|Logs Features| MLMD
        E -.->|Logs Models| MLMD
        E -.->|Logs Metrics| MLflow[(MLflow Server)]
        F -.->|Promotes| MLflow
        MLMD --> Dash[Lineage Dashboard]
    end
    
    subgraph Google Cloud Platform
        G -->|Uploads| GCS[(Cloud Storage)]
        Deploy[deploy.sh] -->|Builds Image| AR[(Artifact Registry)]
        AR --> CR[Cloud Run <br> Prediction API]
        CR -.->|Downloads Model| GCS
    end
```

## Quick Start (Local Pipeline)

1. **Start the environment**:
   ```bash
   cd "Lab 5"
   docker compose up --build -d
   ```
2. **Access URLs**:
   - **Airflow**: http://localhost:8080 (airflow / airflow)
   - **MLflow**: http://localhost:5001
   - **Lineage Dashboard**: http://localhost:5003
3. **Trigger Pipeline**:
   - In Airflow UI, find `manufacturing_defect_detection` and trigger the DAG.
4. **View Lineage**:
   - Open the Lineage Dashboard (Port 5003) to see the interactive MLMD graph representing the pipeline execution.

## GCP Deployment (Serverless API)

To deploy the winning model as a serverless API on Cloud Run:

1. **Authenticate with GCP**:
   ```bash
   gcloud auth login
   ```
2. **Run Deploy Script**:
   ```bash
   ./deploy.sh
   ```
   This script will create a GCS bucket, an Artifact Registry repository, build the lightweight `Dockerfile.serve` image, push it, and deploy it to Cloud Run (`min-instances=0` to stay free).

3. **Make Predictions**:
   Use the Cloud Run URL outputted by the script to send POST requests to `/predict`.

4. **Cleanup**:
   To avoid any accidental charges, run:
   ```bash
   ./cleanup.sh
   ```

## Authors
**Ajith Srikanth** - IE7374 MLOps - Northeastern University
