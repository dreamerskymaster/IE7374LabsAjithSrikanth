import os
import json
import time
import random
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from src.utils import (
    get_logger, log_stage, log_dataframe_summary, log_dict_table,
    FEATURES_PATH, MODEL_PARAM_GRIDS, MLFLOW_EXPERIMENT_NAME, 
    MLFLOW_TRACKING_URI, QUALITY_CLASSES, DATA_DIR, RESULTS_DIR
)

warnings.filterwarnings("ignore")
logger = get_logger(__name__)

def plot_confusion_matrix(y_true, y_pred, model_name, save_path):
    """Generates and saves a confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="RdYlGn_r", 
                xticklabels=QUALITY_CLASSES, yticklabels=QUALITY_CLASSES)
    plt.title(f"Confusion Matrix: {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_feature_importance(model, feature_names, model_name, save_path):
    """Generates and saves a feature importance bar chart."""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(10, 6))
        plt.title(f"Feature Importance: {model_name}")
        plt.barh(range(len(importances)), importances[indices], align="center")
        plt.yticks(range(len(importances)), [feature_names[i] for i in indices])
        plt.xlabel("Relative Importance")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        return dict(zip(feature_names, importances.tolist()))
    return None

@log_stage("Model Training")
def run(model_name, tracker=None, context_id=None, features_artifact_id=None):
    """Trains a specific model with full MLflow tracking."""
    if not FEATURES_PATH.exists():
        logger.error(f"Features file not found at {FEATURES_PATH}")
        return

    df = pd.read_csv(FEATURES_PATH)
    X = df.drop(columns=["quality", "timestamp", "machine_id"], errors="ignore")
    y = df["quality"]
    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    logger.info(f"Training {model_name}...")
    logger.info(f"Train size: {X_train.shape}, Test size: {X_test.shape}")
    
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    try:
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    except Exception as e:
        logger.warning(f"Experiment setup race condition: {e}. Retrying...")
        time.sleep(random.random() * 3)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run(run_name=f"{model_name}_run") as run:
        # Tags
        mlflow.set_tag("model_type", model_name)
        mlflow.set_tag("pipeline_stage", "training")
        mlflow.set_tag("author", "Ajith Srikanth")
        mlflow.set_tag("domain", "Manufacturing")

        # Config & Params
        param_grid = MODEL_PARAM_GRIDS.get(model_name, {})
        mlflow.log_params({"tuning_method": "GridSearchCV"})

        # Model instantiation
        start_train = time.time()
        if model_name == "RandomForest":
            from sklearn.model_selection import GridSearchCV
            base_model = RandomForestClassifier()
        elif model_name == "XGBoost":
            import xgboost as xgb
            from sklearn.model_selection import GridSearchCV
            base_model = xgb.XGBClassifier(eval_metric="mlogloss", use_label_encoder=False)
        elif model_name == "LightGBM":
            import lightgbm as lgb
            from sklearn.model_selection import GridSearchCV
            base_model = lgb.LGBMClassifier()
        elif model_name == "NeuralNet":
            from sklearn.model_selection import GridSearchCV
            base_model = MLPClassifier(validation_fraction=0.15, batch_size=64)
        else:
            raise ValueError(f"Unknown model name: {model_name}")

        # Train with tuning
        logger.info(f"Running GridSearchCV for {model_name}...")
        grid_search = GridSearchCV(base_model, param_grid, cv=3, scoring='f1_weighted', n_jobs=-1, verbose=1)
        grid_search.fit(X_train_scaled, y_train)
        
        model = grid_search.best_estimator_
        train_time = time.time() - start_train
        
        logger.info(f"Best parameters for {model_name}: {grid_search.best_params_}")
        mlflow.log_params(grid_search.best_params_)
        
        # Predict
        y_pred = model.predict(X_test_scaled)
        
        # Metrics
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1_weighted": f1_score(y_test, y_pred, average="weighted"),
            "f1_macro": f1_score(y_test, y_pred, average="macro"),
            "precision_weighted": precision_score(y_test, y_pred, average="weighted"),
            "recall_weighted": recall_score(y_test, y_pred, average="weighted"),
            "training_time_s": train_time
        }
        
        # Per-class F1
        f1_per_class = f1_score(y_test, y_pred, average=None)
        for i, f1 in enumerate(f1_per_class):
            metrics[f"f1_class_{i}"] = f1
            
        mlflow.log_metrics(metrics)
        log_dict_table(logger, metrics, f"{model_name} Metrics")

        # Artifacts
        # 1. Classification Report
        report = classification_report(y_test, y_pred, target_names=QUALITY_CLASSES)
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        report_path = RESULTS_DIR / f"{model_name}_report.txt"
        with open(report_path, "w") as f:
            f.write(report)
        mlflow.log_artifact(str(report_path))

        # 2. Confusion Matrix
        cm_path = RESULTS_DIR / f"{model_name}_cm.png"
        plot_confusion_matrix(y_test, y_pred, model_name, str(cm_path))
        mlflow.log_artifact(str(cm_path))

        # 3. Feature Importance
        fi_path = RESULTS_DIR / f"{model_name}_fi.png"
        fi_data = plot_feature_importance(model, feature_names, model_name, str(fi_path))
        if fi_data:
            mlflow.log_artifact(str(fi_path))
            fi_json_path = RESULTS_DIR / f"{model_name}_fi.json"
            with open(fi_json_path, "w") as f:
                json.dump(fi_data, f, indent=4)
            mlflow.log_artifact(str(fi_json_path))

        # 4. Model
        import joblib
        joblib.dump(model, RESULTS_DIR / f"{model_name}_model.pkl")
        
        if model_name == "XGBoost":
            mlflow.xgboost.log_model(model, "model")
        elif model_name == "LightGBM":
            mlflow.lightgbm.log_model(model, "model")
        else:
            mlflow.sklearn.log_model(model, "model")

        logger.info(f"{model_name} training complete. Run ID: {run.info.run_id}")
        
        # Result dict for evaluator
        result = {
            "model_name": model_name,
            "run_id": run.info.run_id,
            "metrics": metrics
        }
        result_path = RESULTS_DIR / f"{model_name}_result.json"
        with open(result_path, "w") as f:
            json.dump(result, f, indent=4)
        
        if tracker and context_id is not None:
            model_artifact_id = tracker.record_model(
                model_name=model_name,
                accuracy=metrics["accuracy"],
                f1=metrics["f1_weighted"],
                framework="sklearn" if "Net" in model_name or "Forest" in model_name else model_name.lower(),
                path=mlflow.get_artifact_uri("model"), 
                context_id=context_id
            )
            tracker.record_execution(
                exec_type=tracker.training_exec_type,
                properties={"model_type": (model_name, "string"), "training_time_s": (train_time, "double")},
                input_ids=[features_artifact_id] if features_artifact_id else [],
                output_ids=[model_artifact_id],
                context_id=context_id
            )
            return str(result_path), model_artifact_id

        return str(result_path), None

if __name__ == "__main__":
    import sys
    name = sys.argv[1] if len(sys.argv) > 1 else "RandomForest"
    run(name)
