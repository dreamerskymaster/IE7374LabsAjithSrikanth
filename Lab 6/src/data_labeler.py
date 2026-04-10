"""Stage 2: Simulate a data labeling workflow with confidence scores and annotator metadata."""
import numpy as np
import pandas as pd
from datetime import datetime
from src.utils import (
    get_logger, log_stage, log_dataframe_summary, log_dict_table,
    RAW_DATA_PATH, LABELED_DATA_PATH, SENSOR_FEATURES
)

logger = get_logger(__name__)


@log_stage("Data Labeling")
def run():
    """Read raw sensor data, apply rule-based labeling with metadata."""
    df = pd.read_csv(RAW_DATA_PATH)
    logger.info(f"Loaded {len(df)} raw records from {RAW_DATA_PATH}")

    # Compute composite risk score for labeling
    df["risk_score"] = (
        0.30 * (df["vibration"] / 15.0)
        + 0.25 * df["tool_wear"]
        + 0.15 * (df["temperature"] / 350.0)
        + 0.15 * (df["depth_of_cut"] / 5.0)
        + 0.15 * (df["feed_rate"] / 400.0)
    )

    # Assign labels based on risk thresholds (simulating annotator logic)
    conditions = [
        df["risk_score"] < 0.35,
        df["risk_score"] < 0.55,
        df["risk_score"] >= 0.55,
    ]
    labels = ["Good Quality", "Minor Defect", "Major Defect"]
    df["labeled_quality"] = np.select(conditions, labels, default="Good Quality")

    # Simulate annotator metadata
    df["labeler_id"] = "auto-classifier-v1"
    df["label_confidence"] = np.clip(
        1.0 - np.abs(df["risk_score"] - np.where(
            df["labeled_quality"] == "Good Quality", 0.175,
            np.where(df["labeled_quality"] == "Minor Defect", 0.45, 0.70)
        )) * 2.0,
        0.65, 0.99
    ).round(3)
    df["label_timestamp"] = datetime.now().isoformat()

    # Agreement simulation: compare original 'quality' column with labeled_quality
    agreement = (df["quality"] == df["labeled_quality"]).mean()

    # Use the labeled version as the canonical quality column going forward
    df["quality"] = df["labeled_quality"]
    df.drop(columns=["labeled_quality"], inplace=True)

    LABELED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(LABELED_DATA_PATH, index=False)

    # Summary
    label_stats = df["quality"].value_counts().to_dict()
    conf_stats = df.groupby("quality")["label_confidence"].mean().to_dict()

    log_dict_table(logger, label_stats, "Label Distribution")
    log_dict_table(logger, {k: f"{v:.3f}" for k, v in conf_stats.items()}, "Avg Confidence per Class")
    logger.info(f"Inter-annotator agreement (vs raw labels): {agreement:.2%}")
    logger.info(f"Saved {len(df)} labeled records to {LABELED_DATA_PATH}")
    return str(LABELED_DATA_PATH)


if __name__ == "__main__":
    run()
