import os
import logging
import time
import functools
import traceback
from pathlib import Path

# Path Constants
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "cnc_sensor_data.csv"
LABELED_DATA_PATH = DATA_DIR / "labeled" / "cnc_labeled.csv"
FEATURES_PATH = DATA_DIR / "processed" / "cnc_features.csv"
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "champion_model.pkl"
MODEL_META_PATH = MODELS_DIR / "model_metadata.json"
METRICS_PATH = MODELS_DIR / "metrics.json"
FEATURE_SCALER_PATH = MODELS_DIR / "feature_scaler.pkl"
PIPELINE_LOG_PATH = DATA_DIR / "pipeline.log"

# GCP Constants
GCP_PROJECT = os.environ.get("GCP_PROJECT", "ajithmlopsie7374")
GCP_REGION = os.environ.get("GCP_REGION", "us-central1")
GCS_BUCKET = os.environ.get("GCS_BUCKET", "lab6-dvc-ajith")
BQ_DATASET = os.environ.get("BQ_DATASET", "cnc_warehouse")

# Pipeline Version
PIPELINE_VERSION = "lab6-v1.0"

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
    "vibration_temperature_ratio",
    "tool_stress_index",
    "spindle_efficiency",
    "thermal_load",
    "wear_rate"
]

ALL_FEATURES = SENSOR_FEATURES + ENGINEERED_FEATURES

QUALITY_CLASSES = ["Good Quality", "Minor Defect", "Major Defect"]
QUALITY_MAP = {0: "Good Quality", 1: "Minor Defect", 2: "Major Defect"}
QUALITY_REVERSE_MAP = {"Good Quality": 0, "Minor Defect": 1, "Major Defect": 2}

SENSOR_RANGES = {
    "spindle_speed": {"min": 1500, "max": 4500, "unit": "RPM"},
    "feed_rate": {"min": 50, "max": 400, "unit": "mm/min"},
    "depth_of_cut": {"min": 0.5, "max": 5.0, "unit": "mm"},
    "vibration": {"min": 0.5, "max": 15.0, "unit": "mm/s^2"},
    "temperature": {"min": 150, "max": 350, "unit": "C"},
    "tool_wear": {"min": 0.0, "max": 1.0, "unit": "index"}
}


# Logging System
class PipelineFormatter(logging.Formatter):
    """Custom formatter with icons and ANSI colors."""
    grey = "\x1b[38;20m"
    cyan = "\x1b[36;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    green = "\x1b[32;20m"
    reset = "\x1b[0m"

    FORMATS = {
        logging.DEBUG: grey + "[%(asctime)s] %(levelname)s | %(name)s: %(message)s" + reset,
        logging.INFO: cyan + "[%(asctime)s] %(levelname)s | %(name)s: %(message)s" + reset,
        logging.WARNING: yellow + "[%(asctime)s] %(levelname)s | %(name)s: %(message)s" + reset,
        logging.ERROR: red + "[%(asctime)s] %(levelname)s | %(name)s: %(message)s" + reset,
        logging.CRITICAL: bold_red + "[%(asctime)s] %(levelname)s | %(name)s: %(message)s" + reset,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt="%Y-%m-%d %H:%M:%S")
        return formatter.format(record)


def get_logger(name):
    """Creates a configured logger with console and file handlers."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        logger.handlers.clear()

    ch = logging.StreamHandler()
    ch.setFormatter(PipelineFormatter())
    logger.addHandler(ch)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(PIPELINE_LOG_PATH)
    fh.setFormatter(logging.Formatter(
        "[%(asctime)s] %(levelname)s | %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    ))
    logger.addHandler(fh)

    logger.propagate = False
    return logger


def log_stage(stage_name):
    """Decorator that wraps functions with entry/exit banners and timing."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_logger(func.__module__)
            logger.info("=" * 60)
            logger.info(f"STAGE: {stage_name}")
            logger.info("=" * 60)
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                logger.info(f"DONE: {stage_name} completed in {duration:.2f}s")
                return result
            except Exception as e:
                duration = time.time() - start_time
                logger.error(f"FAIL: {stage_name} failed after {duration:.2f}s")
                logger.error(traceback.format_exc())
                raise e
        return wrapper
    return decorator


def log_dataframe_summary(logger, df, name="DataFrame"):
    """Logs key statistics of a pandas DataFrame."""
    logger.info(f"Summary of {name}:")
    logger.info(f"  Shape: {df.shape}")
    logger.info(f"  Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    logger.info(f"  Missing: {df.isnull().sum().sum()}")
    if "quality" in df.columns:
        dist = df["quality"].value_counts().to_dict()
        logger.info(f"  Class distribution: {dist}")


def log_dict_table(logger, data, title="Data"):
    """Formats a dictionary as an aligned table in logs."""
    logger.info(f"--- {title} ---")
    if not data:
        return
    max_key = max(len(str(k)) for k in data.keys())
    for k, v in data.items():
        logger.info(f"  {str(k).ljust(max_key)} | {v}")


def engineer_features_from_raw(df):
    """Applies feature engineering transforms. Used in both pipeline and Cloud Function.
    Input df must have columns: spindle_speed, feed_rate, depth_of_cut, vibration, temperature, tool_wear.
    Returns df with 5 new engineered feature columns appended."""
    df = df.copy()
    df["vibration_temperature_ratio"] = df["vibration"] / (df["temperature"] + 1e-6)
    df["tool_stress_index"] = df["tool_wear"] * df["depth_of_cut"] * df["feed_rate"]
    df["spindle_efficiency"] = df["spindle_speed"] / (df["vibration"] + 1.0)
    df["thermal_load"] = df["temperature"] * df["depth_of_cut"]
    df["wear_rate"] = df["tool_wear"] / (df["spindle_speed"] / 1000.0 + 1e-6)
    return df
