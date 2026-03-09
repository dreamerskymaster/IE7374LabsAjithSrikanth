import os
import json
import mlflow
from datetime import datetime
from src.utils import (
    get_logger, log_stage, log_dict_table,
    DATA_DIR, MLFLOW_EXPERIMENT_NAME, MLFLOW_TRACKING_URI,
    MLFLOW_REGISTRY_MODEL, save_metadata
)

logger = get_logger(__name__)

@log_stage("Model Evaluation & Comparison")
def run_evaluate():
    """Compares all trained models and selects the champion."""
    results = []
    for model_name in ["RandomForest", "XGBoost", "LightGBM", "NeuralNet"]:
        result_path = DATA_DIR / f"{model_name}_result.json"
        if result_path.exists():
            with open(result_path, "r") as f:
                results.append(json.load(f))
        else:
            logger.warning(f"Result file for {model_name} not found at {result_path}")

    if not results:
        logger.error("No model results found to compare.")
        return

    # Sort by weighted F1 score
    results_sorted = sorted(results, key=lambda x: x["metrics"]["f1_weighted"], reverse=True)
    
    comp_data = []
    for r in results_sorted:
        comp_data.append({
            "Model": r["model_name"],
            "Accuracy": f"{r['metrics']['accuracy']:.4f}",
            "F1(wt)": f"{r['metrics']['f1_weighted']:.4f}",
            "F1(mac)": f"{r['metrics']['f1_macro']:.4f}",
            "Precision": f"{r['metrics']['precision_weighted']:.4f}",
            "Recall": f"{r['metrics']['recall_weighted']:.4f}",
            "Time(s)": f"{r['metrics']['training_time_s']:.2f}"
        })
    
    log_dict_table(logger, {d["Model"]: f"F1-wt: {d['F1(wt)']}, Acc: {d['Accuracy']}" for d in comp_data}, "Model Comparison Summary")
    
    champion = results_sorted[0]
    runner_up = results_sorted[1] if len(results_sorted) > 1 else None
    
    logger.info(f"🏆 CHAMPION: {champion['model_name']} (F1-wt: {champion['metrics']['f1_weighted']:.4f})")
    if runner_up:
        margin = champion['metrics']['f1_weighted'] - runner_up['metrics']['f1_weighted']
        logger.info(f"🥈 RUNNER-UP: {runner_up['model_name']} (Margin: {margin:.4f})")

    # Save comparison summary
    summary_path = DATA_DIR / "model_comparison_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "champion": champion,
            "comparison": results_sorted,
            "timestamp": datetime.now().isoformat()
        }, f, indent=4)

    # Update metadata
    save_metadata({
        "champion_model": champion["model_name"],
        "champion_run_id": champion["run_id"],
        "champion_f1_weighted": champion["metrics"]["f1_weighted"]
    })

    return str(summary_path)

@log_stage("Model Registration")
def run_register():
    """Registers the champion model in the MLflow Model Registry."""
    summary_path = DATA_DIR / "model_comparison_summary.json"
    if not summary_path.exists():
        logger.error(f"Comparison summary not found at {summary_path}")
        return

    with open(summary_path, "r") as f:
        summary = json.load(f)
    
    champion = summary["champion"]
    run_id = champion["run_id"]
    model_name = champion["model_name"]
    metrics = champion["metrics"]

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    logger.info(f"Registering {model_name} (Run ID: {run_id}) as '{MLFLOW_REGISTRY_MODEL}'...")
    
    # Register model
    model_uri = f"runs:/{run_id}/model"
    mv = mlflow.register_model(model_uri, MLFLOW_REGISTRY_MODEL)
    
    # Update description and tags
    client = mlflow.tracking.MlflowClient()
    description = (
        f"Champion model: {model_name}\n"
        f"F1 (weighted): {metrics['f1_weighted']:.4f}\n"
        f"Accuracy: {metrics['accuracy']:.4f}\n"
        f"Training Time: {metrics['training_time_s']:.2f}s"
    )
    client.update_model_version(
        name=MLFLOW_REGISTRY_MODEL,
        version=mv.version,
        description=description
    )
    
    # Set tags
    client.set_model_version_tag(MLFLOW_REGISTRY_MODEL, mv.version, "champion_model_type", model_name)
    client.set_model_version_tag(MLFLOW_REGISTRY_MODEL, mv.version, "f1_weighted", f"{metrics['f1_weighted']:.4f}")
    client.set_model_version_tag(MLFLOW_REGISTRY_MODEL, mv.version, "registered_by", "Airflow Pipeline")

    logger.info(f"Successfully registered version {mv.version} of {MLFLOW_REGISTRY_MODEL}")
    logger.info(f"MLflow UI: {MLFLOW_TRACKING_URI}/#/models/{MLFLOW_REGISTRY_MODEL}")

    # Update metadata
    save_metadata({
        "registered_model_name": MLFLOW_REGISTRY_MODEL,
        "registered_version": mv.version
    })

if __name__ == "__main__":
    import sys
    action = sys.argv[1] if len(sys.argv) > 1 else "evaluate"
    if action == "evaluate":
        run_evaluate()
    elif action == "register":
        run_register()
