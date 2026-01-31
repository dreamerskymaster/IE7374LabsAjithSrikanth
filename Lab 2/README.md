# Lab 2: MLOps Foundation - Streamlit & FastAPI Integration

## Project Overview
This laboratory assignment demonstrates the integration of a **FastAPI backend** and a **Streamlit frontend** to deploy a Machine Learning model. The project specifically addresses the classification of Iris flower species using a Random Forest model.

**Student:** Ajith Srikanth  
**Course:** IE7374 - MLOps  

---

## Technical Features
- **FastAPI Inference Engine**: Serves predictions with Pydantic validation.
- **Random Forest Model**: High-accuracy classification trained on morphological data.
- **Streamlit Dashboard**: A premium UI for real-time interaction and file-based batch processing.
- **Self-Documenting Code**: Comprehensive docstrings and architectural comments throughout the codebase.

---

## Directory Layout
- `backend/src/main.py`: The FastAPI application server.
- `backend/src/train.py`: Model training logic.
- `backend/model/`: Storage for the serialized `.pkl` model.
- `frontend/src/Dashboard.py`: The interactive web interface.

---

## Local Deployment Guide

### 1. Initialize Environment
Navigate to the Lab 2 directory and install the necessary libraries:
```bash
pip install -r requirements.txt
```

### 2. Launch the Backend API
From the repository root, execute:
```bash
# Note: Ensure you are in the directory containing 'Lab 2'
uvicorn "Lab 2.backend.src.main:app" --host 0.0.0.0 --port 8000
```

### 3. Launch the Frontend Dashboard
Open a separate terminal and execute:
```bash
streamlit run "Lab 2/frontend/src/Dashboard.py"
```

The dashboard will be accessible via your browser, allowing you to classify Iris species in real-time.
