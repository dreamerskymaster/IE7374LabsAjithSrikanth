import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import os

# --- UI Configuration & Branding ---
# Student: Ajith Srikanth (IE7374 - MLOps)
# Lab 2: Proxy Version (Cloud Optimized)

st.set_page_config(
    page_title="Ajith's Iris Prediction Hub (Cloud)",
    page_icon="🌸",
    layout="centered"
)

# Premium UI Styling
st.markdown("""
    <style>
    .prediction-card {
        padding: 20px;
        border-radius: 10px;
        background-color: #ffffff;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
    .stSlider > div > div > div > div {
        color: #4CAF50;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Core Prediction Logic (Standalone Proxy) ---

def load_prediction_model():
    """
    Service Lifecycle: Model Loader
    Loads the Random Forest classifier directly into the Streamlit context.
    This bypasses the need for an external FastAPI server in the cloud.
    """
    model_path = "iris_model.pkl" # Expects model in the same directory
    if os.path.exists(model_path):
        try:
            with open(model_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            st.error(f"Inference Error: {e}")
            return None
    else:
        st.error(f"Error: Model file '{model_path}' missing from repository.")
        return None

def perform_inference(model, input_data):
    """
    Engine: Direct Prediction
    Takes raw features and returns the botanical classification result.
    """
    species_map = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}
    try:
        data_array = np.array([input_data])
        prediction_idx = model.predict(data_array)[0]
        return species_map.get(int(prediction_idx), "Unknown")
    except Exception as e:
        return f"Processing Error: {str(e)}"

# --- Main Dashboard Application ---

st.title("🌸 Iris species Prediction Hub")
st.caption("A standalone, cloud-optimized deployment for IE7374 MLOps Lab 2.")

# Display a banner image for a more visual experience
try:
    st.image("flower.jpeg", width='stretch', caption="Botanical Exploration System Interface (Cloud Edition)")
except:
    pass

# Pre-load the model for efficiency
model = load_prediction_model()

if model:
    st.success("🚀 Prediction Engine: Fully Operational")
    
    st.divider()
    
    # Interaction Tabs for a cleaner UI
    tab1, tab2 = st.tabs(["Manual sliders", "File Upload"])
    
    with tab1:
        st.subheader("Manual Morphological Entry")
        st.write("Adjust the sliders below to simulate flower measurements.")
        
        c1, c2 = st.columns(2)
        with c1:
            sl = st.slider("Sepal Length", 4.0, 8.0, 5.8, key="sl")
            sw = st.slider("Sepal Width", 2.0, 4.5, 3.0, key="sw")
        with c2:
            pl = st.slider("Petal Length", 1.0, 7.0, 4.3, key="pl")
            pw = st.slider("Petal Width", 0.1, 2.5, 1.3, key="pw")
            
        if st.button("Classify species", type="primary"):
            features = [sl, sw, pl, pw]
            result = perform_inference(model, features)
            st.balloons()
            st.markdown(f"""
                <div class="prediction-card">
                    <h3>Classification Result</h3>
                    <h1 style='color: #4CAF50;'>{result}</h1>
                </div>
            """, unsafe_allow_html=True)
            try:
                st.image("st_sucess.png", width=150)
            except:
                pass

    with tab2:
        st.subheader("Batch JSON processing")
        uploaded_file = st.file_uploader("Upload feature set (JSON format)", type=["json"])
        
        if uploaded_file:
            try:
                data = json.load(uploaded_file)
                st.write("Payload received:")
                st.json(data)
                
                # Extract values in correct order: sepal_length, sepal_width, petal_length, petal_width
                features = [
                    data.get("sepal_length"),
                    data.get("sepal_width"),
                    data.get("petal_length"),
                    data.get("petal_width")
                ]
                
                if all(v is not None for v in features):
                    if st.button("Process Batch Result"):
                        result = perform_inference(model, features)
                        st.info(f"### Predicted species: {result}")
                else:
                    st.warning("JSON structure mismatch. Expected keys: sepal_length, sepal_width, petal_length, petal_width")
            except Exception as e:
                st.error(f"File Parse Error: {e}")

else:
    st.error("⚠️ System Failure: Inference engine could not be initialized.")

# Footer Section
st.divider()
st.caption("Developed by Ajith Srikanth | Prof. Ramin | Northeastern University | IE7374 MLOps")
