# Lab 3: Manufacturing Quality Prediction with Docker Compose

> **CNC Machining Defect Prediction** — A high-fidelity, 3-stage containerized ML pipeline for predicting precision quality outcomes from manufacturing sensor telemetry.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.2-orange)

---

## Overview

This lab elevates the multi-container pattern with a **production-grade** manufacturing use case and strict industrial safety standards:

| Feature | Lab 2 (Iris) | Lab 3 (Precision Manufacturing) |
|---|---|---|
| **Domain** | Iris flower classification | CNC machining defect prediction |
| **Services** | 2 (train → serve) | 3 (generate → train → serve) |
| **Data Volume** | 150 records | 5000 high-fidelity sensor records |
| **ML Framework** | TensorFlow/Keras | scikit-learn (RandomForest) |
| **Model Specs** | Simple Neural Network | Optimized RandomForest (200 trees, Depth 15) |
| **Accuracy** | Baseline | **85.7% (Tuned for Industrial Precision)** |
| **Input Safety** | Basic types | **Strict Server-side Range Validation** |
| **Terminology** | Mixed | **Full Terminology (No Shortforms)** |
| **Monitoring** | None | `/health` + `/metrics` endpoints |
| **Security** | Root user | Non-root user with structured logging |

---

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  data-generator  │────▶│  model-training  │────▶│    serving      │
│                 │     │                 │     │                 │
│  Generates 5000 │     │  RandomForest   │     │  Premium API    │
│  Sensor Records │     │  (200 Estimators)│    │  + Glassmorphic │
│  w/ Fail Rules  │     │  85.7% Accuracy │     │  Port 5001      │
└────────┬────────┘     └────────┬────────┘     └────────┬────────┘
         │                       │                       │
         └───────────────────────┴───────────────────────┘
                         pipeline_data (shared volume)
```

### Industrial Sensor Telemetry
- **Spindle Speed** — Revolutions Per Minute
- **Feed Rate** — Millimeters Per Minute
- **Depth of Cut** — Millimeters
- **Vibration Amplitude** — Millimeters Per Second
- **Process Temperature** — Degrees Celsius
- **Cutting Tool Wear** — Millimeters

### Operational Outcomes
- 🟢 **Good Quality** — Process within nominal tolerances
- 🟡 **Minor Defect** — Non-critical surface or dimensional variance
- 🔴 **Major Defect** — Critical failure requiring scrap or rework

---

## How to Run

```bash
# Build and execute the full cycle
docker compose up --build

# Run in detached mode for production testing
docker compose up --build -d
```

Once running, access the dashboard at **http://localhost:5001**.

### Predictive API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Premium Dashboard UI |
| `/predict` | POST | High-fidelity prediction engine (strict validation) |
| `/health` | GET | Real-time service status check |
| `/metrics` | GET | Model accuracy and operational metrics |

### Professional Validation (curl)

```bash
curl -X POST http://localhost:5001/predict \
  -H "Content-Type: application/json" \
  -d '{"spindle_speed": 3500, "feed_rate": 450, "depth_of_cut": 2.5, "vibration": 5.2, "temperature": 180, "tool_wear": 0.15}'
```

---

## Cleanup

```bash
docker compose down -v   # Terminate containers and purge shared volumes
```

---

## Project Hierarchy

```
Lab 3/
├── docker-compose.yml        # Multi-service orchestration & volume mapping
├── Dockerfile                # High-security production image definition (Non-root)
├── requirements.txt          # Explicit version-locked dependencies
├── .dockerignore             # Efficient build-context management
├── README.md                 # Technical documentation
└── src/
    ├── data_generator.py     # Stage 1: Industrial telemetry generation (5000 samples)
    ├── model_training.py     # Stage 2: Hyperparameter-tuned RandomForest pipeline
    ├── main.py               # Stage 3: Strict-validation serving API
    └── templates/
        └── predict.html      # Premium glassmorphic analytics dashboard
```

---

**Author:** Ajith Srikanth | MLOps & Industrial AI — Docker Labs
