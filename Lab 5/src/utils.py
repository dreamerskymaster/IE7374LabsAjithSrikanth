import os
import json
import logging
import time
import functools
import traceback
from datetime import datetime
from pathlib import Path
import pandas as pd

# Path Constants
DATA_DIR = Path(os.getenv("DATA_DIR", "/opt/airflow/data"))
RAW_DATA_PATH = DATA_DIR / "raw_sensor_data.csv"
VALIDATED_DATA_PATH = DATA_DIR / "validated_sensor_data.csv"
FEATURES_PATH = DATA_DIR / "engineered_features.csv"
METADATA_PATH = DATA_DIR / "pipeline_metadata.json"
PIPELINE_LOG_PATH = DATA_DIR / "pipeline_execution.log"
RESULTS_DIR = DATA_DIR / "results"
MODELS_DIR = DATA_DIR / "models"

MLMD_DB_PATH = os.environ.get("MLMD_DB_PATH", "/opt/airflow/mlmd/mlmd.sqlite")
GCS_BUCKET = os.environ.get("GCS_BUCKET", "")
GCP_PROJECT = os.environ.get("GCP_PROJECT", "ajithmlopsie7374")
TFDV_STATS_PATH = str(DATA_DIR / "tfdv_stats.pb")
TFDV_SCHEMA_PATH = str(DATA_DIR / "tfdv_schema.pb")
TFDV_ANOMALIES_PATH = str(DATA_DIR / "tfdv_anomalies.pb")
TFDV_STATS_HTML = str(DATA_DIR / "tfdv_stats.html")
PIPELINE_VERSION = "lab5-v1.0"
# MLflow Config
MLFLOW_EXPERIMENT_NAME = "manufacturing-defect-detection"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5001")
MLFLOW_REGISTRY_MODEL = "manufacturing-defect-champion"

# Manufacturing Constants
SENSOR_FEATURES = [
    "spindle_speed", 
    "feed_rate", 
    "depth_of_cut", 
    "vibration", 
    "temperature", 
    "tool_wear"
]

ENGINEERED_FEATURES = [
    "speed_feed_ratio", 
    "vibration_temp_interaction", 
    "tool_wear_severity", 
    "cutting_energy_proxy", 
    "thermal_stress_index"
]

QUALITY_CLASSES = ["Good", "Minor Defect", "Major Defect"]
QUALITY_MAP = {0: "Good", 1: "Minor Defect", 2: "Major Defect"}

SENSOR_RANGES = {
    "spindle_speed": {"min": 800, "max": 3200, "unit": "RPM"},
    "feed_rate": {"min": 50, "max": 400, "unit": "mm/min"},
    "depth_of_cut": {"min": 0.5, "max": 4.0, "unit": "mm"},
    "vibration": {"min": 0.1, "max": 8.0, "unit": "mm/s^2"},
    "temperature": {"min": 150, "max": 350, "unit": "C"},
    "tool_wear": {"min": 0.0, "max": 1.0, "unit": "index"}
}

# Model Hyperparameter Grids for Tuning
MODEL_PARAM_GRIDS = {
    "RandomForest": {
        "n_estimators": [100, 200, 300],
        "max_depth": [8, 12, 16],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "class_weight": ["balanced"],
        "random_state": [42]
    },
    "XGBoost": {
        "n_estimators": [100, 200],
        "max_depth": [4, 6, 8],
        "learning_rate": [0.01, 0.1, 0.2],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
        "random_state": [42]
    },
    "LightGBM": {
        "n_estimators": [100, 200],
        "max_depth": [6, 10, -1],
        "learning_rate": [0.01, 0.1],
        "num_leaves": [31, 63],
        "class_weight": ["balanced"],
        "verbose": [-1],
        "random_state": [42]
    },
    "NeuralNet": {
        "hidden_layer_sizes": [(64, 32), (128, 64, 32)],
        "activation": ["relu", "tanh"],
        "solver": ["adam"],
        "learning_rate_init": [0.001, 0.01],
        "max_iter": [500],
        "early_stopping": [True],
        "random_state": [42]
    }
}

# Logging System
class PipelineFormatter(logging.Formatter):
    """Custom formatter for console output with icons and colors."""
    grey = "\x1b[38;20m"
    cyan = "\x1b[36;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    green = "\x1b[32;20m"
    reset = "\x1b[0m"
    
    FORMATS = {
        logging.DEBUG: grey + "🔍 [%(asctime)s] %(levelname)s - %(name)s: %(message)s" + reset,
        logging.INFO: cyan + "✅ [%(asctime)s] %(levelname)s - %(name)s: %(message)s" + reset,
        logging.WARNING: yellow + "⚠️ [%(asctime)s] %(levelname)s - %(name)s: %(message)s" + reset,
        logging.ERROR: red + "❌ [%(asctime)s] %(levelname)s - %(name)s: %(message)s" + reset,
        logging.CRITICAL: bold_red + "🚨 [%(asctime)s] %(levelname)s - %(name)s: %(message)s" + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        # Add OK icon for specific info messages if desired
        formatter = logging.Formatter(log_fmt, datefmt="%Y-%m-%d %H:%M:%S")
        return formatter.format(record)

class FileFormatter(logging.Formatter):
    """Plain text formatter for log files."""
    def __init__(self):
        super().__init__("[%Y-%m-%d %H:%M:%S] %(levelname)s - %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

def get_logger(name):
    """Creates a configured logger with console and file handlers."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    if logger.hasHandlers():
        logger.handlers.clear()
        
    # Console Handler
    ch = logging.StreamHandler()
    ch.setFormatter(PipelineFormatter())
    logger.addHandler(ch)
    
    # File Handler
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(PIPELINE_LOG_PATH)
    fh.setFormatter(FileFormatter())
    logger.addHandler(fh)
    
    logger.propagate = False
    return logger

# Decorators
def log_stage(stage_name):
    """Wraps functions with entry/exit banners and timing."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_logger(func.__module__)
            logger.info("="*60)
            logger.info(f"STAGE: {stage_name}")
            logger.info("="*60)
            
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                logger.info(f"SUCCESS: {stage_name} completed in {duration:.2f}s")
                return result
            except Exception as e:
                duration = time.time() - start_time
                logger.error(f"FAILURE: {stage_name} failed after {duration:.2f}s")
                logger.error(traceback.format_exc())
                raise e
        return wrapper
    return decorator

# Metadata I/O
def save_metadata(data, path=METADATA_PATH):
    """Saves pipeline metadata to a JSON file."""
    try:
        existing_data = load_metadata(path)
        existing_data.update(data)
        existing_data["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        with open(path, 'w') as f:
            json.dump(existing_data, f, indent=4)
    except Exception as e:
        print(f"Error saving metadata: {e}")

def load_metadata(path=METADATA_PATH):
    """Loads pipeline metadata from a JSON file."""
    if not path.exists():
        return {}
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception:
        return {}

# Display Helpers
def log_dataframe_summary(logger, df, name="DataFrame"):
    """Logs key statistics of a pandas DataFrame."""
    logger.info(f"Summary of {name}:")
    logger.info(f" - Shape: {df.shape}")
    logger.info(f" - Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    logger.info(f" - Missing Values: {df.isnull().sum().sum()}")
    if "quality" in df.columns:
        dist = df["quality"].value_counts().to_dict()
        logger.info(f" - Class Distribution: {dist}")

def log_dict_table(logger, data, title="Data"):
    """Formats a dictionary as an aligned table in logs."""
    logger.info(f"--- {title} ---")
    if not data:
        return
    max_key = max(len(str(k)) for k in data.keys())
    for k, v in data.items():
        logger.info(f" {str(k).ljust(max_key)} | {v}")
