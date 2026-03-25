import pandas as pd
import numpy as np
from src.utils import (
    get_logger, log_stage, log_dataframe_summary, log_dict_table,
    RAW_DATA_PATH, VALIDATED_DATA_PATH, SENSOR_FEATURES, SENSOR_RANGES,
    save_metadata
)

logger = get_logger(__name__)

@log_stage("Data Validation")
def run():
    """Performs 5 validation checks on the raw sensor data."""
    if not RAW_DATA_PATH.exists():
        logger.error(f"Raw data file not found at {RAW_DATA_PATH}")
        return

    df = pd.read_csv(RAW_DATA_PATH)
    initial_rows = len(df)
    logger.info(f"Starting validation on {initial_rows} rows")

    # 1. Schema check
    REQUIRED_COLUMNS = SENSOR_FEATURES + ["quality"]
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    logger.info(f"Check 1/5: Schema validation")
    if missing_cols:
        logger.error(f"Missing columns: {missing_cols}")
        raise ValueError(f"Schema validation failed: Missing {missing_cols}")
    logger.info(" - All required columns present")

    # 2. Target integrity
    logger.info(f"Check 2/5: Target integrity")
    invalid_quality = df[~df["quality"].isin([0, 1, 2])]
    if not invalid_quality.empty:
        logger.warning(f"Found {len(invalid_quality)} invalid quality labels. Dropping.")
        df = df[df["quality"].isin([0, 1, 2])]
    else:
        logger.info(" - Quality labels are valid {0, 1, 2}")

    # 3. Range validation
    logger.info(f"Check 3/5: Range validation")
    range_issues = 0
    for feature, ranges in SENSOR_RANGES.items():
        out_of_range = df[(df[feature] < ranges["min"]) | (df[feature] > ranges["max"])]
        if not out_of_range.empty:
            logger.warning(f" - {feature}: {len(out_of_range)} values out of range [{ranges['min']}, {ranges['max']}] {ranges['unit']}")
            range_issues += len(out_of_range)
    
    if range_issues == 0:
        logger.info(" - All sensors within expected ranges")
    else:
        logger.info(f" - Total range issues detected: {range_issues}")

    # 4. Null check
    logger.info(f"Check 4/5: Null values check")
    null_counts = df[REQUIRED_COLUMNS].isnull().sum()
    total_nulls = null_counts.sum()
    if total_nulls > 0:
        logger.warning(f"Found {total_nulls} null values. Dropping rows.")
        df = df.dropna(subset=REQUIRED_COLUMNS)
    else:
        logger.info(" - No null values detected in sensor or quality columns")

    # 5. Duplicate detection
    logger.info(f"Check 5/5: Duplicate detection")
    duplicates = df.duplicated(subset=SENSOR_FEATURES).sum()
    if duplicates > 0:
        logger.warning(f"Found {duplicates} duplicate sensor readings. Removing.")
        df = df.drop_duplicates(subset=SENSOR_FEATURES)
    else:
        logger.info(" - No exact duplicate sensor readings found")

    # Conclusion
    final_rows = len(df)
    dropped_rows = initial_rows - final_rows
    drop_rate = (dropped_rows / initial_rows) * 100

    validation_stats = {
        "Input Rows": initial_rows,
        "Dropped Rows": dropped_rows,
        "Output Rows": final_rows,
        "Drop Rate %": f"{drop_rate:.2f}%"
    }
    log_dict_table(logger, validation_stats, "Validation Results")

    if drop_rate > 10:
        logger.warning(f"High drop rate detected: {drop_rate:.2f}%!")

    log_dataframe_summary(logger, df, "Validated Data")

    # Save output
    df.to_csv(VALIDATED_DATA_PATH, index=False)
    logger.info(f"Validated data saved to {VALIDATED_DATA_PATH}")

    # Update metadata
    metadata = {
        "validation_stats": {
            "initial_rows": initial_rows,
            "final_rows": final_rows,
            "dropped_rows": dropped_rows,
            "drop_rate_pct": drop_rate,
            "validation_timestamp": pd.Timestamp.now().isoformat()
        }
    }
    save_metadata(metadata)

if __name__ == "__main__":
    run()
