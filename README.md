# IE7374 - MLOps Laboratory Assignments

**Student:** Ajith Srikanth  
**Course:** IE7374 - MLOps  
**Professor:** Prof. Ramin Mohammadi  
**Institution:** Northeastern University  

---

## 📖 Repository Overview
This repository serves as a comprehensive portfolio of laboratory assignments for the IE7374 MLOps course. Each module is designed to demonstrate proficiency in Python development, infrastructure automation, UI/UX design, and Machine Learning integration.

---

## 📂 Laboratory Modules

### [Lab 1: Foundation - Calculator & CI/CD](./Lab%201)
*The architecture of reliability.*
- **Core**: Advanced mathematical logic (`Power`, `Square Root`, `Average`).
- **Safety**: 100% code coverage across Pytest and Unittest frameworks.
- **Automation**: Dual-pipeline GitHub Actions for continuous quality assurance.
- **Setup**: Isolated virtual environment structure.

### [Lab 2: Integration - Streamlit & FastAPI](./Lab%202)
*The intersection of Engineering and Science.*
- **Backend**: High-performance FastAPI server serving a Random Forest model.
- **Frontend**: Premium Streamlit dashboard with real-time botanical classification.
- **Features**: Batch JSON processing and manual feature manipulation.
- **Documentation**: Self-explanatory, production-grade codebase with signature comments.

### [Lab 2 Proxy: Standalone Cloud Version](./Lab%202%20Proxy)
*Optimized for the Cloud.*
- **Live Demo**: [lab2flowerprediction.streamlit.app](https://lab2flowerprediction.streamlit.app/)
- **Architecture**: Simplified, unified deployment model.
- **Platform**: Designed for seamless hosting on **Streamlit Cloud**.
- **Efficiency**: Direct model integration for low-latency, serverless execution.

### [Lab 3: Full Stack ML App](./Lab%203)
*End-to-end Machine Learning Web Application.*
- **Backend**: FastAPI with PostgreSQL database.
- **Frontend**: Premium Jinja2/HTML/CSS dashboard.
- **Orchestration**: Docker Compose for multi-container deployment.
- **Features**: Authentication, model prediction, and historical tracking.

### [Lab 4: MLOps Pipeline](./Lab%204)
*Manufacturing Defect Detection with Airflow & MLflow.*
- **Orchestration**: Apache Airflow DAGs.
- **Experiment Tracking**: MLflow multi-model training (RF, XGB, LGB, NN).
- **Automation**: Champion model auto-registration based on F1 score.
- **Infrastructure**: Containerized with 2GB memory limits for robust execution.

### [Lab 5: MLOps Defect Pipeline on GCP](./Lab%205)
*Production-grade tracking, profiling, and cloud deployment.*
- **Data Profiling**: TensorFlow Data Validation (TFDV) for auto-schema generation and anomaly detection.
- **Lineage Tracking**: Google ML Metadata (MLMD) with a custom interactive dashboard visualization.
- **Advanced Tuning**: Extended hyperparameter grid searching via `GridSearchCV`.
- **Cloud Deployment**: Serverless prediction API on GCP Cloud Run with zero-cost tier configuration.

---

## 🚀 Quick Start Guide

### Lab 1 (Testing)
```bash
cd "Lab 1"
pip install -r requirements.txt
pytest test/test_pytest.py
```

### Lab 2 (Integrated API & Dashboard)
1. **API**: `uvicorn "Lab 2.backend.src.main:app" --port 8000`
2. **UI**: `streamlit run "Lab 2/frontend/src/Dashboard.py"`

### Lab 2 Proxy (Standalone)
```bash
cd "Lab 2 Proxy"
pip install -r requirements.txt
streamlit run app.py
```

### Lab 3 (Full Stack App)
```bash
cd "Lab 3"
docker compose up --build -d
```
Access at http://localhost:8000

### Lab 4 (MLOps Pipeline)
```bash
cd "Lab 4"
docker compose up --build -d
```
Access Airflow at http://localhost:8080 and MLflow at http://localhost:5001

### Lab 5 (GCP & Metadata Pipeline)
```bash
cd "Lab 5"
docker compose up --build -d
```
Access Airflow at http://localhost:8080, MLflow at http://localhost:5001, and the Lineage Dashboard at http://localhost:5003.

**Serverless API Deployment:**
```bash
./deploy.sh
```

---

## 📜 Documentation & Attribution
All implementations feature unique architectural styles, comprehensive docstrings, and premium styling curated by **Ajith Srikanth**.

Copyright © 2026 Ajith Srikanth. Distributed under the IE7374 Course Guidelines.
