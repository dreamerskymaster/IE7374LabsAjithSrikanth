from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import sys
import os

# Ensure src modules are importable
sys.path.append("/opt/airflow")

from src import (
    data_generator,
    data_validator_tfdv as data_validator,
    feature_engineering,
    model_trainer,
    model_evaluator
)

def get_tracker_and_context(kwargs):
    from src.mlmd_tracker import PipelineTracker
    tracker = PipelineTracker()
    run_id = kwargs['dag_run'].run_id
    context_id = tracker.get_or_create_pipeline_run(run_id)
    return tracker, context_id

def task_generate_data(**kwargs):
    t, c = get_tracker_and_context(kwargs)
    return data_generator.run(tracker=t, context_id=c)

def task_validate_data(**kwargs):
    t, c = get_tracker_and_context(kwargs)
    return data_validator.run(tracker=t, context_id=c)

def task_feature_engineering(**kwargs):
    t, c = get_tracker_and_context(kwargs)
    return feature_engineering.run(tracker=t, context_id=c)

def task_train_model(model_name, **kwargs):
    t, c = get_tracker_and_context(kwargs)
    return model_trainer.run(model_name=model_name, tracker=t, context_id=c)

def task_evaluate(**kwargs):
    t, c = get_tracker_and_context(kwargs)
    return model_evaluator.run_evaluate(tracker=t, context_id=c)

def task_register(**kwargs):
    t, c = get_tracker_and_context(kwargs)
    return model_evaluator.run_register(tracker=t, context_id=c)

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
        python_callable=task_generate_data,
        doc_md="Generates 3000 synthetic sensor readings."
    )

    val_data = PythonOperator(
        task_id="validate_data",
        python_callable=task_validate_data,
        doc_md="Validates sensor ranges and schema integrity."
    )

    feat_eng = PythonOperator(
        task_id="feature_engineering",
        python_callable=task_feature_engineering,
        doc_md="Calculates engineered features like Cutting Energy and Tool Wear Severity."
    )

    evaluate = PythonOperator(
        task_id="evaluate_and_compare",
        python_callable=task_evaluate,
        doc_md="Compares models and saves champion metadata."
    )

    register = PythonOperator(
        task_id="register_best_model",
        python_callable=task_register,
        doc_md="Registers the champion model in MLflow Model Registry."
    )

    # Parallel Training Tasks
    models = ["RandomForest", "XGBoost", "LightGBM", "NeuralNet"]
    training_tasks = []

    for model_name in models:
        task_id = f"train_{model_name.lower().replace(' ', '_')}"
        train_task = PythonOperator(
            task_id=task_id,
            python_callable=task_train_model,
            op_args=[model_name],
            doc_md=f"Trains {model_name} with full MLflow tracking."
        )
        training_tasks.append(train_task)

    # Dependencies
    gen_data >> val_data >> feat_eng >> training_tasks >> evaluate >> register
