import streamlit as st
import requests
import pandas as pd
import json
import pickle
import numpy as np
import os

# --- UI Configuration & Styling ---
# Author: Ajith Srikanth (IE7374 - MLOps)
# Project: Iris botanical Classification Dashboard

st.set_page_config(
    page_title="Ajith's Iris Prediction Dashboard",
    page_icon="🌸",
    layout="wide"
)

# Custom CSS for a more premium, high-fidelity look
st.markdown("""
    <style>
    /* Gradient background for the entire app */
    .stApp {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    /* Elegant button styling */
    .stButton>button {
        width: 100%;
        border-radius: 12px;
        height: 3.5em;
        background: linear-gradient(90deg, #4CAF50, #45a049);
        color: white;
        font-weight: bold;
        border: none;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
        background: linear-gradient(90deg, #45a049, #4CAF50);
    }

    /* Sidebar sophistication */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #dee2e6;
    }

    /* Card-like containers for inputs */
    div.stSlider {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Backend Integration Configuration ---
BACKEND_URL = "http://localhost:8000"

def check_backend_status():
    """
    Utility: connectivity Check
    Verifies if the FastAPI backend is operational.
    """
    try:
        response = requests.get(f"{BACKEND_URL}/")
        return response.status_code == 200
    except:
        return False

def load_standalone_model():
    """
    Fallback: Standalone Inference Engine
    Loads the model directly if the FastAPI backend is unreachable.
    """
    # Try different possible paths based on execution context
    paths = [
        "Lab 2 Proxy/backend/model/iris_model.pkl",
        "backend/model/iris_model.pkl",
        "../backend/model/iris_model.pkl"
    ]
    for p in paths:
        if os.path.exists(p):
            try:
                with open(p, 'rb') as f:
                    return pickle.load(f)
            except:
                continue
    return None

def perform_prediction(features, model=None):
    """
    Prediction Orchestrator
    Attempts API prediction first, falls back to standalone if model is loaded.
    """
    # 1. Try API first
    try:
        payload = {
            "sepal_length": features[0],
            "sepal_width": features[1],
            "petal_length": features[2],
            "petal_width": features[3]
        }
        response = requests.post(f"{BACKEND_URL}/predict", json=payload, timeout=2)
        if response.status_code == 200:
            return response.json().get("prediction"), "API"
    except:
        pass
    
    # 2. Fallback to standalone model
    if model:
        species_map = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}
        pred = model.predict(np.array([features]))[0]
        return species_map.get(int(pred), "Unknown"), "Standalone"
    
    return None, None

# --- dashboard Header ---
st.title("🌿 🌸 Iris Botanical Prediction & Exploration 🌸 🌿")

# Display a banner image for a more visual experience
try:
    st.image("Lab 2 Proxy/frontend/assets/flower.jpeg", width='stretch', caption="Botanical Exploration System Interface")
except:
    pass

st.markdown("""
### ✨ Welcome to the Professional Artificial Intelligence Botanical Interface! ✨
This application bridges the gap between **Machine Learning** and **Botanical Science** 🧪🔬. 
Using a highly-trained Random Forest architecture, we can identify Iris species with incredible precision 🎯.
""")

# --- Sidebar: System Status & Input controls ---
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Check if backend is alive
    is_online = check_backend_status()
    standalone_model = load_standalone_model()
    
    if is_online:
        st.success("✅ Backend API: Online")
    elif standalone_model:
        st.warning("⚡ Backend: Standalone Mode")
        st.caption("API unreachable. Using embedded model for cloud compatibility.")
    else:
        st.error("❌ Backend API: Offline")
        st.info("Ensure the FastAPI server is running on port 8000.")

    st.divider()
    st.subheader("📥 Input Method Selection")
    input_mode = st.radio("Choose interaction style:", ["Slider Manipulation 🎚️", "Bulk Data Upload 📂"])

# --- Main Interaction Area ---
if input_mode == "Manual Sliders":
    st.subheader("🔢 Manual Feature entry")
    col1, col2 = st.columns(2)
    
    with col1:
        sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.8, step=0.1)
        sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.0, step=0.1)
    
    with col2:
        petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 4.3, step=0.1)
        petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 1.3, step=0.1)

    if st.button("🔍 Run Prediction"):
        features = [sepal_length, sepal_width, petal_length, petal_width]
        
        with st.spinner("Analyzing botanical signatures..."):
            prediction, provider = perform_prediction(features, standalone_model)
            
            if prediction:
                st.balloons()
                st.success(f"### 🎉 Result Identified: **{prediction}**")
                st.caption(f"🛡️ Security Logic: Processed via {provider} Engine")
                # Removed st_sucess.png asset display as requested
            else:
                st.error("Prediction failed: Backend unreachable and local model missing.")

else:
    st.subheader("📂 Batch Prediction (JSON)")
    uploaded_file = st.file_uploader("Upload an Iris feature JSON file", type=["json"])
    
    if uploaded_file is not None:
        try:
            data = json.load(uploaded_file)
            st.write("File loaded successfully. Data preview:")
            st.json(data)
            
            if st.button("🚀 Process Batch Prediction"):
                # Extract features from JSON (mirroring API logic)
                try:
                    features = [
                        data.get("sepal_length"),
                        data.get("sepal_width"),
                        data.get("petal_length"),
                        data.get("petal_width")
                    ]
                    
                    if all(v is not None for v in features):
                        prediction, provider = perform_prediction(features, standalone_model)
                        if prediction:
                            st.success(f"### Predicted species: **{prediction}**")
                            st.caption(f"Processed via: {provider} Engine")
                        else:
                            st.error("Batch processing failed.")
                    else:
                        st.warning("JSON structure mismatch. Expected: sepal_length, sepal_width, petal_length, petal_width")
                except Exception as e:
                    st.error(f"Processing Error: {e}")
        except Exception as e:
            st.error(f"Error parsing file: {e}")

# --- Footer ---
st.divider()
st.caption("✨ Developed with excellence by **Ajith Srikanth** | IE7374 MLOps 🎓 | Professor Ramin | Northeastern University 🐾")
