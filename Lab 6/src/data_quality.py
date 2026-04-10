"""Stage 5: Run data quality checks and write report to BigQuery."""
import pandas as pd
from datetime import datetime
from src.utils import (
    get_logger, log_stage,
    RAW_DATA_PATH, SENSOR_RANGES, GCP_PROJECT, BQ_DATASET
)

logger = get_logger(__name__)


def check_nulls(df):
    """Check for null values."""
    nulls = df.isnull().sum().to_dict()
    total = sum(nulls.values())
    return {"check": "null_values", "passed": total == 0, "detail": f"{total} nulls found", "counts": nulls}


def check_ranges(df):
    """Check sensor values are within expected ranges."""
    violations = {}
    for col, bounds in SENSOR_RANGES.items():
        if col in df.columns:
            out = ((df[col] < bounds["min"]) | (df[col] > bounds["max"])).sum()
            if out > 0:
                violations[col] = int(out)
    passed = len(violations) == 0
    return {"check": "range_validation", "passed": passed, "detail": f"{sum(violations.values())} violations" if violations else "All in range", "violations": violations}


def check_duplicates(df):
    """Check for duplicate record_ids."""
    if "record_id" not in df.columns:
        return {"check": "duplicates", "passed": True, "detail": "No record_id column"}
    dupes = df["record_id"].duplicated().sum()
    return {"check": "duplicates", "passed": dupes == 0, "detail": f"{dupes} duplicates"}


def check_class_balance(df):
    """Check class distribution is not extremely skewed."""
    if "quality" not in df.columns:
        return {"check": "class_balance", "passed": True, "detail": "No quality column"}
    dist = df["quality"].value_counts(normalize=True).to_dict()
    min_pct = min(dist.values())
    return {"check": "class_balance", "passed": min_pct > 0.05, "detail": f"Min class: {min_pct:.1%}", "distribution": {k: round(v, 4) for k, v in dist.items()}}


@log_stage("Data Quality Checks")
def run():
    """Run all quality checks and optionally write to BigQuery."""
    df = pd.read_csv(RAW_DATA_PATH)
    logger.info(f"Running quality checks on {len(df)} records")

    checks = [check_nulls(df), check_ranges(df), check_duplicates(df), check_class_balance(df)]

    all_passed = all(c["passed"] for c in checks)
    for c in checks:
        status = "PASS" if c["passed"] else "FAIL"
        logger.info(f"  [{status}] {c['check']}: {c['detail']}")

    # Attempt to write to BigQuery
    try:
        from google.cloud import bigquery
        client = bigquery.Client(project=GCP_PROJECT)
        table_id = f"{GCP_PROJECT}.{BQ_DATASET}.data_quality_report"

        rows = []
        for c in checks:
            rows.append({
                "check_name": c["check"],
                "passed": c["passed"],
                "detail": c["detail"],
                "run_timestamp": datetime.now().isoformat(),
                "record_count": len(df),
            })
        report_df = pd.DataFrame(rows)
        report_df["run_timestamp"] = pd.to_datetime(report_df["run_timestamp"])
        job = client.load_table_from_dataframe(report_df, table_id)
        job.result()
        logger.info(f"Quality report written to BigQuery: {table_id}")
    except Exception as e:
        logger.warning(f"Could not write to BigQuery (skipping): {e}")

    logger.info(f"Overall quality: {'ALL PASSED' if all_passed else 'SOME CHECKS FAILED'}")
    return all_passed


if __name__ == "__main__":
    run()
