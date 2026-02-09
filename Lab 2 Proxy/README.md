# Lab 2 Proxy: Standalone Cloud-Ready Iris Prediction

## Project Overview
This is a standalone, simplified version of Lab 2 designed specifically for **Streamlit Cloud** deployment. It consolidates the machine learning logic and the interactive dashboard into a single application environment, removing the dependency on a local FastAPI server for easier cloud hosting.

**Student:** Ajith Srikanth  
**Course:** IE7374 - MLOps  

---

## Technical Features
- **Integrated Inference Engine**: The Random Forest model is loaded directly into the Streamlit session.
- **Unified Architecture**: Single-folder deployment (`Lab 2 Proxy/`) containing all necessary logic and assets.
- **Premium UI**: Enhanced with botanical visuals and custom CSS styling for a sophisticated user experience.
- **Signature Documentation**: Detailed inline documentation following Ajith's unique architectural commenting style.

---

## Directory Layout
- `app.py`: The main entry point for the standalone application.
- `iris_model.pkl`: The serialized Random Forest classifier.
- `requirements.txt`: Cloud-optimized dependency list.
- `flower.jpeg`: Integrated botanical banner.
- `st_sucess.png`: Visual feedback asset.

---

## How to Deploy (Streamlit Cloud)
1.  Connect this GitHub repository to your Streamlit Cloud account.
2.  Set the **Main file path** to: `Lab 2 Proxy/app.py`.
3.  Deploy! The environment will automatically detect the configuration in `Lab 2 Proxy/requirements.txt`.

---

## Local Execution
If you wish to run the standalone version locally:
```bash
# Navigate to Lab 2 Proxy
cd "Lab 2 Proxy"

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```
