from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import sys
import os

# Ensure src modules are importable
sys.path.append("/opt/airflow")

from src import (
    data_generator,
    data_validator,
    feature_engineering,
    model_trainer,
    model_evaluator
)

default_args = {
    "owner": "Ajith Srikanth",
    "depends_on_past": False,
    "start_date": datetime(2024, 1, 1),
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    "manufacturing_defect_detection",
    default_args=default_args,
    description="MLOps Pipeline for CNC Manufacturing Defect Detection",
    schedule_interval=None,
    catchup=False,
    tags=["manufacturing", "mlflow", "defect-detection", "lab4"],
    doc_md="""
# Manufacturing Defect Detection Pipeline (Lab 4)
This pipeline orchestrates the end-to-end ML lifecycle for predicting machining quality.

## Stages:
1. **Generate Data**: Synthetic sensor data creation with physics rules.
2. **Validate Data**: 5-point data quality check.
3. **Feature Engineering**: Domain-specific feature extraction.
4. **Parallel Training**: RF, XGBoost, LightGBM, and NeuralNet training with MLflow.
5. **Evaluate & Compare**: champion selection based on F1 score.
6. **Register Model**: Promotes the champion to MLflow Model Registry.

**Author:** Ajith Srikanth
    """
) as dag:

    gen_data = PythonOperator(
        task_id="generate_data",
        python_callable=data_generator.run,
        doc_md="Generates 3000 synthetic sensor readings."
    )

    val_data = PythonOperator(
        task_id="validate_data",
        python_callable=data_validator.run,
        doc_md="Validates sensor ranges and schema integrity."
    )

    feat_eng = PythonOperator(
        task_id="feature_engineering",
        python_callable=feature_engineering.run,
        doc_md="Calculates engineered features like Cutting Energy and Tool Wear Severity."
    )

    evaluate = PythonOperator(
        task_id="evaluate_and_compare",
        python_callable=model_evaluator.run_evaluate,
        doc_md="Compares models and saves champion metadata."
    )

    register = PythonOperator(
        task_id="register_best_model",
        python_callable=model_evaluator.run_register,
        doc_md="Registers the champion model in MLflow Model Registry."
    )

    # Parallel Training Tasks
    models = ["RandomForest", "XGBoost", "LightGBM", "NeuralNet"]
    training_tasks = []

    for model_name in models:
        task_id = f"train_{model_name.lower().replace(' ', '_')}"
        train_task = PythonOperator(
            task_id=task_id,
            python_callable=model_trainer.run,
            op_args=[model_name],
            doc_md=f"Trains {model_name} with full MLflow tracking."
        )
        training_tasks.append(train_task)

    # Dependencies
    gen_data >> val_data >> feat_eng >> training_tasks >> evaluate >> register
