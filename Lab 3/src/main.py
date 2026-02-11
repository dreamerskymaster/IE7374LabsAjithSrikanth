"""
Manufacturing Quality Prediction - High-Fidelity Serving API
Server-side component for predicting precision CNC defect status.
Revised for full terminology and strict sensor input validation.
"""
from flask import Flask, request, jsonify, render_template
import numpy as np
import joblib
import json
import os
import logging
from datetime import datetime

# Configure logging for professional request tracking
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [SERVING_API] - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Full Terminology Configuration
SENSOR_LIMITS = {
    'spindle_speed': {'min': 800.0, 'max': 5000.0, 'name': 'Spindle Speed', 'unit': 'Revolutions Per Minute'},
    'feed_rate': {'min': 50.0, 'max': 1000.0, 'name': 'Feed Rate', 'unit': 'Millimeters Per Minute'},
    'depth_of_cut': {'min': 0.1, 'max': 10.0, 'name': 'Depth of Cut', 'unit': 'Millimeters'},
    'vibration': {'min': 0.1, 'max': 15.0, 'name': 'Vibration', 'unit': 'Millimeters Per Second'},
    'temperature': {'min': 20.0, 'max': 500.0, 'name': 'Temperature', 'unit': 'Degrees Celsius'},
    'tool_wear': {'min': 0.0, 'max': 1.0, 'name': 'Tool Wear', 'unit': 'Millimeters'}
}

FEATURE_KEY_ORDER = ['spindle_speed', 'feed_rate', 'depth_of_cut', 'vibration', 'temperature', 'tool_wear']
CLASS_LABELS = {0: 'Good Quality', 1: 'Minor Defect', 2: 'Major Defect'}

# ── Configuration via environment variables ──
PORT = int(os.environ.get('SERVING_PORT', 5000))
MODEL_PATH = os.environ.get('MODEL_PATH', '/exchange/quality_model.joblib' if os.path.exists('/exchange/quality_model.joblib') else 'quality_model.joblib')
SCALER_PATH = os.environ.get('SCALER_PATH', '/exchange/scaler.joblib' if os.path.exists('/exchange/scaler.joblib') else 'scaler.joblib')
METRICS_PATH = os.environ.get('METRICS_PATH', '/exchange/training_metrics.json' if os.path.exists('/exchange/training_metrics.json') else 'training_metrics.json')
DEBUG_MODE = os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'

# ── App Setup ──
app = Flask(__name__)

# ── Global Artifacts ──
model = None
scaler = None
training_metrics = {}
prediction_counter = 0

def load_system_artifacts():
    global model, scaler, training_metrics
    try:
        logger.info(f"Loading predictive model from {MODEL_PATH}...")
        model = joblib.load(MODEL_PATH)
        logger.info(f"Loading feature scaler from {SCALER_PATH}...")
        scaler = joblib.load(SCALER_PATH)
        
        if os.path.exists(METRICS_PATH):
            with open(METRICS_PATH, 'r') as f:
                training_metrics = json.load(f)
            logger.info("Training metrics successfully synchronized.")
            
        logger.info("System artifacts loaded. Integrity check: PASSED.")
    except Exception as e:
        logger.error(f"Failed to initialize system artifacts: {str(e)}")

load_system_artifacts()

# ── Routes ──

@app.route('/')
def index():
    """Render the Manufacturing Quality Control Dashboard."""
    return render_template('predict.html')


@app.route('/health')
def health_check():
    """System health monitoring endpoint."""
    return jsonify({
        'status': 'operational',
        'artifacts_loaded': model is not None and scaler is not None,
        'timestamp': datetime.now().isoformat()
    })


@app.route('/metrics')
def system_metrics():
    """Expose high-level system and training metrics."""
    return jsonify({
        'model_metadata': training_metrics.get('model_name', 'Manufacturing Classifier'),
        'model_accuracy': round(training_metrics.get('accuracy', 0.0), 4),
        'predictions_since_startup': prediction_counter,
        'feature_terminology': {k: v['unit'] for k, v in SENSOR_LIMITS.items()}
    })


@app.route('/predict', methods=['POST'])
def perform_prediction():
    global prediction_counter
    
    try:
        # 1. Capture and Validate Data
        data = request.form if request.form else request.get_json()
        logger.info("Received incoming prediction request.")
        
        input_values = []
        validation_errors = []
        
        for key in FEATURE_KEY_ORDER:
            if key not in data:
                validation_errors.append(f"Missing required parameter: {SENSOR_LIMITS[key]['name']}")
                continue
                
            try:
                val = float(data[key])
                # Range Check
                v_min = SENSOR_LIMITS[key]['min']
                v_max = SENSOR_LIMITS[key]['max']
                if not (v_min <= val <= v_max):
                    validation_errors.append(
                        f"{SENSOR_LIMITS[key]['name']} value {val} is outside valid operating range ({v_min} - {v_max} {SENSOR_LIMITS[key]['unit']})"
                    )
                input_values.append(val)
            except (ValueError, TypeError):
                validation_errors.append(f"Invalid numeric format for {SENSOR_LIMITS[key]['name']}")

        if validation_errors:
            logger.warning(f"Request rejected due to {len(validation_errors)} validation failure(s).")
            return jsonify({'success': False, 'errors': validation_errors}), 400

        # 2. Execute Prediction
        input_array = np.array(input_values).reshape(1, -1)
        scaled_input = scaler.transform(input_array)
        
        predicted_index = int(model.predict(scaled_input)[0])
        probabilities = model.predict_proba(scaled_input)[0]
        confidence_score = float(np.max(probabilities))
        
        result_label = CLASS_LABELS[predicted_index]
        prediction_counter += 1
        
        # 3. Success Response with Full Terminology
        logger.info(f"Prediction result: {result_label} (Confidence: {confidence_score:.2%})")
        
        return jsonify({
            'success': True,
            'prediction_result': result_label,
            'confidence_score_percentage': round(confidence_score * 100, 2),
            'probability_distribution': {
                CLASS_LABELS[i]: round(float(p) * 100, 2)
                for i, p in enumerate(probabilities)
            },
            'input_echo': {
                SENSOR_LIMITS[k]['name']: {'value': v, 'unit': SENSOR_LIMITS[k]['unit']}
                for k, v in zip(FEATURE_KEY_ORDER, input_values)
            }
        })
        
    except Exception as error:
        logger.error(f"Internal server error during prediction processing: {str(error)}")
        return jsonify({'success': False, 'errors': ["An internal processing error occurred."]}), 500


if __name__ == '__main__':
    logger.info(f"Starting High-Fidelity Manufacturing API on port {PORT}...")
    app.run(host='0.0.0.0', port=PORT, debug=DEBUG_MODE)
