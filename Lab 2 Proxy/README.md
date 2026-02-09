# Lab 2 Proxy: MLOps Foundation Mirror

## Project Overview
This folder is an **exact mirrored copy** of Lab 2. It contains the full integrated source code for both the **FastAPI backend** and the **Streamlit frontend**. This version is designated for deployment testing and redundant hosting on platforms like Streamlit Cloud.

**Student:** Ajith Srikanth  
**Course:** IE7374 - MLOps  

---

## Technical Features
- **FastAPI Inference Hub**: Production-grade backend serving the Iris classification model.
- **Random Forest Model**: Trained and serialized classifier for flower species.
- **Enhanced Streamlit Interface**: Premium UI with botanical visuals (`flower.jpeg`).
- **Integrated Assets**: All images and state icons included for a complete experience.

---

## Directory Layout
- `backend/src/main.py`: The FastAPI application server.
- `backend/src/train.py`: Model training logic.
- `backend/model/`: Serialization directory for the model.
- `frontend/src/Dashboard.py`: The interactive Streamlit dashboard.
- `frontend/assets/`: UI assets library.

---

## Deployment & Hosting (Streamlit Cloud)
To host this mirrored version:
1.  Connect this GitHub repository to Streamlit Cloud.
2.  Set the **Main file path** to: `Lab 2 Proxy/frontend/src/Dashboard.py`.
3.  Deploy! The system will use the configuration within this directory.

---

## Local Execution
```bash
# Start Backend
uvicorn "Lab 2 Proxy.backend.src.main:app" --port 8000

# Start Frontend
streamlit run "Lab 2 Proxy/frontend/src/Dashboard.py"
```
