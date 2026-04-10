"""Stage 6: Load data into BigQuery warehouse with proper schemas and views."""
import pandas as pd
from src.utils import (
    get_logger, log_stage, log_dict_table,
    RAW_DATA_PATH, LABELED_DATA_PATH, GCP_PROJECT, BQ_DATASET
)

logger = get_logger(__name__)

# BigQuery table schemas
SCHEMAS = {
    "sensor_readings": [
        {"name": "record_id", "type": "STRING"},
        {"name": "batch_id", "type": "STRING"},
        {"name": "machine_id", "type": "STRING"},
        {"name": "timestamp", "type": "TIMESTAMP"},
        {"name": "spindle_speed", "type": "FLOAT64"},
        {"name": "feed_rate", "type": "FLOAT64"},
        {"name": "depth_of_cut", "type": "FLOAT64"},
        {"name": "vibration", "type": "FLOAT64"},
        {"name": "temperature", "type": "FLOAT64"},
        {"name": "tool_wear", "type": "FLOAT64"},
        {"name": "quality", "type": "STRING"},
    ],
    "defect_labels": [
        {"name": "record_id", "type": "STRING"},
        {"name": "quality", "type": "STRING"},
        {"name": "risk_score", "type": "FLOAT64"},
        {"name": "labeler_id", "type": "STRING"},
        {"name": "label_confidence", "type": "FLOAT64"},
        {"name": "label_timestamp", "type": "TIMESTAMP"},
    ],
    "prediction_log": [
        {"name": "prediction_id", "type": "STRING"},
        {"name": "timestamp", "type": "TIMESTAMP"},
        {"name": "spindle_speed", "type": "FLOAT64"},
        {"name": "feed_rate", "type": "FLOAT64"},
        {"name": "depth_of_cut", "type": "FLOAT64"},
        {"name": "vibration", "type": "FLOAT64"},
        {"name": "temperature", "type": "FLOAT64"},
        {"name": "tool_wear", "type": "FLOAT64"},
        {"name": "predicted_class", "type": "STRING"},
        {"name": "confidence", "type": "FLOAT64"},
        {"name": "model_version", "type": "STRING"},
    ],
    "data_quality_report": [
        {"name": "check_name", "type": "STRING"},
        {"name": "passed", "type": "BOOLEAN"},
        {"name": "detail", "type": "STRING"},
        {"name": "run_timestamp", "type": "TIMESTAMP"},
        {"name": "record_count", "type": "INT64"},
    ],
}

# SQL views
VIEWS = {
    "v_defect_rate_by_machine": f"""
        CREATE OR REPLACE VIEW `{GCP_PROJECT}.{BQ_DATASET}.v_defect_rate_by_machine` AS
        SELECT
            machine_id,
            COUNT(*) AS total_readings,
            COUNTIF(quality != 'Good Quality') AS defect_count,
            ROUND(COUNTIF(quality != 'Good Quality') / COUNT(*) * 100, 2) AS defect_rate_pct
        FROM `{GCP_PROJECT}.{BQ_DATASET}.sensor_readings`
        GROUP BY machine_id
        ORDER BY defect_rate_pct DESC
    """,
    "v_daily_quality_trend": f"""
        CREATE OR REPLACE VIEW `{GCP_PROJECT}.{BQ_DATASET}.v_daily_quality_trend` AS
        SELECT
            DATE(timestamp) AS production_date,
            quality,
            COUNT(*) AS count
        FROM `{GCP_PROJECT}.{BQ_DATASET}.sensor_readings`
        GROUP BY production_date, quality
        ORDER BY production_date
    """,
    "v_high_risk_readings": f"""
        CREATE OR REPLACE VIEW `{GCP_PROJECT}.{BQ_DATASET}.v_high_risk_readings` AS
        SELECT *
        FROM `{GCP_PROJECT}.{BQ_DATASET}.defect_labels`
        WHERE risk_score > 0.6
        ORDER BY risk_score DESC
    """,
}


@log_stage("Warehouse Loading")
def run():
    """Create BigQuery tables and load data from CSV files."""
    try:
        from google.cloud import bigquery
    except ImportError:
        logger.error("google-cloud-bigquery not installed. Run: pip install google-cloud-bigquery")
        return

    client = bigquery.Client(project=GCP_PROJECT)
    dataset_ref = f"{GCP_PROJECT}.{BQ_DATASET}"

    # Create dataset if needed
    try:
        client.get_dataset(dataset_ref)
        logger.info(f"Dataset {BQ_DATASET} exists")
    except Exception:
        from google.cloud.bigquery import Dataset
        ds = Dataset(dataset_ref)
        ds.location = "US"
        ds.description = "CNC Manufacturing Data Warehouse (Lab 6)"
        client.create_dataset(ds)
        logger.info(f"Created dataset {BQ_DATASET}")

    # Create tables with schemas
    for table_name, schema_fields in SCHEMAS.items():
        table_id = f"{dataset_ref}.{table_name}"
        from google.cloud.bigquery import Table, SchemaField
        schema = [SchemaField(f["name"], f["type"]) for f in schema_fields]
        table = Table(table_id, schema=schema)
        try:
            client.create_table(table, exists_ok=True)
            logger.info(f"Table ready: {table_name}")
        except Exception as e:
            logger.warning(f"Table {table_name}: {e}")

    # Load sensor_readings from raw CSV
    raw_df = pd.read_csv(RAW_DATA_PATH)
    raw_df["timestamp"] = pd.to_datetime(raw_df["timestamp"])
    table_id = f"{dataset_ref}.sensor_readings"
    job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
    job = client.load_table_from_dataframe(raw_df, table_id, job_config=job_config)
    job.result()
    logger.info(f"Loaded {len(raw_df)} rows into sensor_readings")

    # Load defect_labels from labeled CSV
    labeled_df = pd.read_csv(LABELED_DATA_PATH)
    label_cols = ["record_id", "quality", "risk_score", "labeler_id", "label_confidence", "label_timestamp"]
    available = [c for c in label_cols if c in labeled_df.columns]
    label_out = labeled_df[available].copy()
    if "label_timestamp" in label_out.columns:
        label_out["label_timestamp"] = pd.to_datetime(label_out["label_timestamp"])
    table_id = f"{dataset_ref}.defect_labels"
    job = client.load_table_from_dataframe(label_out, table_id, job_config=job_config)
    job.result()
    logger.info(f"Loaded {len(label_out)} rows into defect_labels")

    # Create views
    for view_name, view_sql in VIEWS.items():
        try:
            client.query(view_sql).result()
            logger.info(f"View created: {view_name}")
        except Exception as e:
            logger.warning(f"View {view_name}: {e}")

    # Summary
    loaded = {"sensor_readings": len(raw_df), "defect_labels": len(label_out)}
    log_dict_table(logger, loaded, "Rows Loaded")
    logger.info("Warehouse loading complete")


if __name__ == "__main__":
    run()
