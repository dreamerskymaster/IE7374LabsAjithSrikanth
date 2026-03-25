import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils import (
    get_logger, log_stage, log_dataframe_summary, log_dict_table,
    VALIDATED_DATA_PATH, FEATURES_PATH, ENGINEERED_FEATURES,
    save_metadata
)

logger = get_logger(__name__)

@log_stage("Feature Engineering")
def run(tracker=None, context_id=None, input_artifact_id=None):
    """Creates domain-specific features and analyzes correlations."""
    if not VALIDATED_DATA_PATH.exists():
        logger.error(f"Validated data file not found at {VALIDATED_DATA_PATH}")
        return

    df = pd.read_csv(VALIDATED_DATA_PATH)
    logger.info(f"Input data shape: {df.shape}")

    # 1. speed_feed_ratio
    logger.info("Creating speed_feed_ratio = spindle_speed / (feed_rate + 1e-6)")
    df["speed_feed_ratio"] = df["spindle_speed"] / (df["feed_rate"] + 1e-6)

    # 2. vibration_temp_interaction
    logger.info("Creating vibration_temp_interaction = vibration * temperature / 1000")
    df["vibration_temp_interaction"] = df["vibration"] * df["temperature"] / 1000

    # 3. tool_wear_severity
    logger.info("Creating tool_wear_severity = tool_wear^2 * vibration")
    df["tool_wear_severity"] = (df["tool_wear"] ** 2) * df["vibration"]

    # 4. cutting_energy_proxy
    logger.info("Creating cutting_energy_proxy = spindle_speed * feed_rate * depth_of_cut / 1e6")
    df["cutting_energy_proxy"] = (df["spindle_speed"] * df["feed_rate"] * df["depth_of_cut"]) / 1e6

    # 5. thermal_stress_index
    logger.info("Creating thermal_stress_index = ((temperature - 200) / 100 * depth_of_cut).clip(0)")
    df["thermal_stress_index"] = (((df["temperature"] - 200) / 100) * df["depth_of_cut"]).clip(0)

    # Feature Statistics
    feat_stats = df[ENGINEERED_FEATURES].describe().T[["mean", "std", "min", "max"]].to_dict("index")
    stats_table = {k: f"Mean: {v['mean']:.2f}, Std: {v['std']:.2f}, Range: [{v['min']:.2f}, {v['max']:.2f}]" 
                   for k, v in feat_stats.items()}
    log_dict_table(logger, stats_table, "Engineered Feature Statistics")

    # Correlation Analysis
    logger.info("Calculating feature-target correlations...")
    correlations = df[ENGINEERED_FEATURES + ["quality"]].corr()["quality"].drop("quality").abs().sort_values(ascending=False)
    
    corr_table = {}
    for feat, val in correlations.items():
        bar = "█" * int(val * 20)
        corr_table[feat] = f"{val:.4f} {bar}"
    log_dict_table(logger, corr_table, "Feature Correlations with Quality")

    # Final Logging
    log_dataframe_summary(logger, df, "Features Ready for Training")
    df.to_csv(FEATURES_PATH, index=False)
    logger.info(f"Features saved to {FEATURES_PATH} ({FEATURES_PATH.stat().st_size / 1024:.2f} KB)")

    # Update metadata
    metadata = {
        "feature_engineering_stats": {
            "total_features": len(df.columns),
            "engineered_features": ENGINEERED_FEATURES,
            "top_correlations": correlations.to_dict(),
            "timestamp": pd.Timestamp.now().isoformat()
        }
    }
    save_metadata(metadata)

    if tracker and context_id is not None:
        features_artifact_id = tracker.record_artifact(
            type_id=tracker.feature_set_type,
            uri=str(FEATURES_PATH),
            properties={"path": (str(FEATURES_PATH), "string"), "n_features": (len(df.columns), "int"), "n_samples": (len(df), "int")},
            context_id=context_id
        )
        tracker.record_execution(
            exec_type=tracker.feature_exec_type,
            properties={"n_features": (len(ENGINEERED_FEATURES), "int")},
            input_ids=[input_artifact_id] if input_artifact_id else [],
            output_ids=[features_artifact_id],
            context_id=context_id
        )
        return str(FEATURES_PATH), features_artifact_id
        
    return str(FEATURES_PATH), None

if __name__ == "__main__":
    run()
