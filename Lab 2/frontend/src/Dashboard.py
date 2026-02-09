import streamlit as st
import requests
import pandas as pd
import json

# --- UI Configuration & Styling ---
# Author: Ajith Srikanth (IE7374 - MLOps)
# Project: Iris botanical Classification Dashboard

st.set_page_config(
    page_title="Ajith's Iris Prediction Dashboard",
    page_icon="🌸",
    layout="wide"
)

# Custom CSS for a more premium look
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #4CAF50;
        color: white;
    }
    .status-box {
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 20px;
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

# --- dashboard Header ---
st.title("🌸 Iris Flower species classification")

# Display a banner image for a more visual experience
try:
    st.image("Lab 2/frontend/assets/dashboard.png", width='stretch', caption="Botanical Exploration System Interface")
except:
    pass

st.markdown("""
Welcome to the **IE7374 MLOps Iris Prediction Dashboard**. 
This application interfaces with a FastAPI backend serving a Random Forest model trained on morphological flower data.
""")

# --- Sidebar: System Status & Input controls ---
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Check if backend is alive
    is_online = check_backend_status()
    if is_online:
        st.success("✅ Backend API: Online")
    else:
        st.error("❌ Backend API: Offline")
        st.info("Ensure the FastAPI server is running on port 8000.")

    st.divider()
    st.subheader("Input Method")
    input_mode = st.radio("Choose interaction style:", ["Manual Sliders", "Bulk JSON Upload"])

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
        if is_online:
            payload = {
                "sepal_length": sepal_length,
                "sepal_width": sepal_width,
                "petal_length": petal_length,
                "petal_width": petal_width
            }
            
            with st.spinner("Analyzing botanical signatures..."):
                response = requests.post(f"{BACKEND_URL}/predict", json=payload)
                
                if response.status_code == 200:
                    prediction = response.json().get("prediction")
                    st.balloons()
                    
                    # Visually pleasing result presentation
                    st.success(f"### Prediction Result: **{prediction}**")
                    try:
                        st.image("Lab 2/frontend/assets/st_sucess.png", width=200)
                    except:
                        pass
                else:
                    st.error(f"Prediction Error: {response.text}")
        else:
            st.warning("Prediction aborted: Backend is unreachable.")

else:
    st.subheader("📂 Batch Prediction (JSON)")
    uploaded_file = st.file_uploader("Upload an Iris feature JSON file", type=["json"])
    
    if uploaded_file is not None:
        try:
            data = json.load(uploaded_file)
            st.write("File loaded successfully. Data preview:")
            st.json(data)
            
            if st.button("🚀 Process Batch Prediction"):
                if is_online:
                    response = requests.post(f"{BACKEND_URL}/predict", json=data)
                    if response.status_code == 200:
                        prediction = response.json().get("prediction")
                        st.success(f"### Predicted species: **{prediction}**")
                    else:
                        st.error("Invalid JSON structure for Iris API.")
                else:
                    st.warning("Backend offline.")
        except Exception as e:
            st.error(f"Error parsing file: {e}")

# --- Footer ---
st.divider()
st.caption("Developed by Ajith Srikanth | IE7374 MLOps | Professor Ramin | Northeastern University")
