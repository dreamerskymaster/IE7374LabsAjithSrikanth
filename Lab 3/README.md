# Lab 3: Manufacturing Quality Prediction with Docker Compose

> **CNC Machining Defect Prediction** — A 3-stage containerized ML pipeline for predicting product quality from manufacturing sensor data.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.2-orange)

---

## Overview

This lab extends the multi-container pattern from Lab 2 with a **manufacturing-focused** use case and several architectural improvements:

| Feature | Lab 2 (Iris) | Lab 3 (Manufacturing) |
|---|---|---|
| **Domain** | Iris flower classification | CNC machining defect prediction |
| **Services** | 2 (train → serve) | 3 (generate → train → serve) |
| **ML Framework** | TensorFlow/Keras | scikit-learn (RandomForest) |
| **Model Format** | .keras | .joblib (with scaler) |
| **Health Checks** | None | `/health` endpoint + Docker HEALTHCHECK |
| **Monitoring** | None | `/metrics` endpoint with training stats |
| **Predictions** | Class only | Class + confidence + probability breakdown |
| **Security** | Root user | Non-root user in Dockerfile |
| **Config** | Hardcoded | Environment variables |
| **Resource Limits** | None | Memory limits per service |

---

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  data-generator  │────▶│  model-training  │────▶│    serving      │
│                 │     │                 │     │                 │
│  Generates 2000 │     │  RandomForest   │     │  Flask API      │
│  CNC sensor     │     │  Classifier     │     │  + Web UI       │
│  records        │     │  (150 trees)    │     │  Port 5000      │
└────────┬────────┘     └────────┬────────┘     └────────┬────────┘
         │                       │                       │
         └───────────────────────┴───────────────────────┘
                         pipeline_data (shared volume)
```

### Sensor Features
- **Spindle Speed** (RPM) — Rotational speed of cutting tool
- **Feed Rate** (mm/min) — Workpiece travel speed
- **Depth of Cut** (mm) — Cutting depth into material
- **Vibration** (mm/s) — Machine vibration amplitude
- **Temperature** (°C) — Cutting zone temperature
- **Tool Wear** (mm) — Flank wear on cutting tool

### Quality Classes
- 🟢 **Good Quality** — Within all tolerances
- 🟡 **Minor Defect** — Surface roughness out of spec
- 🔴 **Major Defect** — Dimensional tolerance failure

---

## How to Run

```bash
# Build and start all 3 services
docker compose up --build

# Or run in detached mode
docker compose up --build -d
```

Once running, open **http://localhost:5001** in your browser.

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web UI for predictions |
| `/predict` | POST | JSON/form prediction endpoint |
| `/health` | GET | Container health check |
| `/metrics` | GET | Training metrics & serving stats |

### Example Prediction (curl)

```bash
curl -X POST http://localhost:5001/predict \
  -d "spindle_speed=2500&feed_rate=200&depth_of_cut=2.0&vibration=6.5&temperature=290&tool_wear=0.45"
```

---

## Cleanup

```bash
docker compose down -v   # Stop containers and remove volume
```

---

## File Structure

```
Lab3/
├── docker-compose.yml        # 3-service pipeline orchestration
├── Dockerfile                # Production-ready image definition
├── requirements.txt          # Python dependencies
├── .dockerignore             # Build context exclusions
├── README.md                 # This file
└── src/
    ├── data_generator.py     # Stage 1: Synthetic CNC data generation
    ├── model_training.py     # Stage 2: RandomForest training pipeline
    ├── main.py               # Stage 3: Flask serving API
    └── templates/
        └── predict.html      # Manufacturing-themed web UI
```

---

**Author:** Ajith Srikanth | MLOps — Docker Labs
