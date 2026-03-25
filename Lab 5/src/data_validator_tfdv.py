import os
import json
import pandas as pd
from utils import (
    get_logger, log_stage, RAW_DATA_PATH, VALIDATED_DATA_PATH, SENSOR_RANGES,
    TFDV_STATS_PATH, TFDV_SCHEMA_PATH, TFDV_ANOMALIES_PATH, TFDV_STATS_HTML
)

try:
    import tensorflow_data_validation as tfdv
    TFDV_AVAILABLE = True
except ImportError:
    TFDV_AVAILABLE = False

logger = get_logger(__name__)

def generate_custom_stats(df):
    stats = []
    for col in df.columns:
        col_stats = {
            "feature": col,
            "type": str(df[col].dtype),
            "count": len(df[col]),
            "missing": int(df[col].isnull().sum()),
            "unique": int(df[col].nunique())
        }
        if pd.api.types.is_numeric_dtype(df[col]):
            col_stats.update({
                "mean": float(df[col].mean()),
                "std": float(df[col].std()),
                "min": float(df[col].min()),
                "max": float(df[col].max())
            })
        stats.append(col_stats)
    return stats

@log_stage("Data Validation (TFDV)")
def run_validation(tracker=None, context_id=None):
    logger.info(f"Loading raw data from {RAW_DATA_PATH}")
    if not os.path.exists(RAW_DATA_PATH):
        raise FileNotFoundError(f"Raw data not found at {RAW_DATA_PATH}")
        
    df = pd.read_csv(RAW_DATA_PATH)
    initial_rows = len(df)
    
    # 1. Basic fallback validation
    logger.info("Running basic range validations...")
    for col, ranges in SENSOR_RANGES.items():
        if col in df.columns:
            out_of_bounds = df[(df[col] < ranges["min"]) | (df[col] > ranges["max"])]
            if not out_of_bounds.empty:
                logger.warning(f"Found {len(out_of_bounds)} anomalous values for {col}")
    
    df = df.dropna()
    df = df.drop_duplicates()
    dropped = initial_rows - len(df)
    
    df.to_csv(VALIDATED_DATA_PATH, index=False)
    logger.info(f"Basic validation complete. Dropped {dropped} rows. Saved to {VALIDATED_DATA_PATH}")
    
    anomalies_found = 0
    tfdv_used = 0
    
    # 2. TFDV Validation
    if TFDV_AVAILABLE:
        logger.info("TFDV is available. Generating statistics and schema...")
        tfdv_used = 1
        stats = tfdv.generate_statistics_from_csv(str(RAW_DATA_PATH))
        
        # Save stats
        logger.info(f"Writing stats to {TFDV_STATS_PATH}")
        tfdv.write_stats_text(stats, TFDV_STATS_PATH)
        
        # Infer Schema
        schema = tfdv.infer_schema(stats)
        logger.info(f"Writing schema to {TFDV_SCHEMA_PATH}")
        tfdv.write_schema_text(schema, TFDV_SCHEMA_PATH)
        
        # Validate Anomalies
        anomalies = tfdv.validate_statistics(stats, schema)
        logger.info(f"Writing anomalies to {TFDV_ANOMALIES_PATH}")
        tfdv.write_anomalies_text(anomalies, TFDV_ANOMALIES_PATH)
        
        # Generate HTML
        try:
            logger.info("Generating TFDV HTML report...")
            # We can use tfdv.get_statistics_html to get the raw HTML
            html = tfdv.get_statistics_html(stats)
            with open(TFDV_STATS_HTML, "w") as f:
                f.write(html)
        except Exception as e:
            logger.error(f"Failed to generate TFDV HTML: {e}")
            
        anomalies_found = len(anomalies.anomaly_info)
        logger.info(f"TFDV detected {anomalies_found} schema anomalies.")
        
    else:
        logger.warning("TFDV not available, using custom statistics generator")
        custom_stats = generate_custom_stats(df)
        with open(TFDV_STATS_PATH.replace('.pb', '.json'), 'w') as f:
            json.dump(custom_stats, f, indent=2)
            
        # Basic HTML table for lineage dashboard
        html = "<h2>Custom Pipeline Statistics</h2><table border='1'><tr><th>Feature</th><th>Count</th><th>Missing</th><th>Mean</th><th>Min</th><th>Max</th></tr>"
        for s in custom_stats:
            html += f"<tr><td>{s['feature']}</td><td>{s['count']}</td><td>{s['missing']}</td>"
            html += f"<td>{s.get('mean', 'N/A')}</td><td>{s.get('min', 'N/A')}</td><td>{s.get('max', 'N/A')}</td></tr>"
        html += "</table>"
        with open(TFDV_STATS_HTML, "w") as f:
            f.write(html)
    
    validated_artifact_id = None
    if tracker and context_id is not None:
        validated_artifact_id = tracker.record_validation(
            path=str(VALIDATED_DATA_PATH),
            n_samples=len(df),
            anomalies_found=anomalies_found,
            context_id=context_id
        )
        # Also could record TFDV stats/schema artifacts here, but for brevity tracking the main one
        # And execution
        tracker.record_execution(
            exec_type=tracker.validation_exec_type,
            properties={
                "checks_passed": (len(df), "int"),
                "rows_dropped": (dropped, "int"),
                "tfdv_used": (tfdv_used, "int")
            },
            input_ids=[], # ideally raw_artifact_id, but the task DAG passes only context_id
            output_ids=[validated_artifact_id] if validated_artifact_id else [],
            context_id=context_id
        )

    return VALIDATED_DATA_PATH, validated_artifact_id

if __name__ == "__main__":
    run_validation()
