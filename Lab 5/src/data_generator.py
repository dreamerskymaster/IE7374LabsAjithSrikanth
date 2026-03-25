import numpy as np
import pandas as pd
import random
from src.utils import (
    get_logger, log_stage, log_dataframe_summary, log_dict_table,
    RAW_DATA_PATH, DATA_DIR, SENSOR_FEATURES, save_metadata
)

logger = get_logger(__name__)

@log_stage("Data Generation")
def run(tracker=None, context_id=None):
    """Generates synthetic CNC sensor readings with defect labels."""
    num_samples = 3000
    np.random.seed(42)
    random.seed(42)

    # 1. Sensor distributions
    data = {
        "spindle_speed": np.clip(np.random.normal(2000, 500, num_samples), 800, 3200),
        "feed_rate": np.clip(np.random.normal(200, 80, num_samples), 50, 400),
        "depth_of_cut": np.random.uniform(0.5, 4.0, num_samples),
        "vibration": np.clip(np.random.exponential(2.5, num_samples), 0.5, 8.0),
        "temperature": np.clip(np.random.normal(240, 45, num_samples), 150, 350),
        "tool_wear": np.random.beta(2, 5, num_samples) * 0.8
    }

    df = pd.DataFrame(data)

    # 2. Defect labeling rules
    def get_quality(row):
        # Major Defect (label=2)
        if (row["vibration"] > 5.5 and row["tool_wear"] > 0.45) or \
           (row["temperature"] > 310 and row["depth_of_cut"] > 3.0) or \
           (row["spindle_speed"] > 2800 and row["feed_rate"] > 330 and row["vibration"] > 4.0):
            return 2
        
        # Minor Defect (label=1)
        if (row["vibration"] > 3.5 and row["tool_wear"] > 0.30) or \
           (row["temperature"] > 280 and row["depth_of_cut"] > 2.5) or \
           (row["feed_rate"] > 300 and row["vibration"] > 3.0) or \
           (row["tool_wear"] > 0.50):
            return 1
            
        return 0

    df["quality"] = df.apply(get_quality, axis=1)

    # 3. Rules statistics logging
    rule_stats = {
        "Major Defect Rules": len(df[df["quality"] == 2]),
        "Minor Defect Rules": len(df[df["quality"] == 1]),
        "Good Samples": len(df[df["quality"] == 0])
    }
    log_dict_table(logger, rule_stats, "Baseline Quality Distribution")

    # 4. Inject 3% label noise
    noise_pct = 0.03
    num_noise = int(num_samples * noise_pct)
    noise_indices = np.random.choice(df.index, num_noise, replace=False)
    for idx in noise_indices:
        df.at[idx, "quality"] = random.choice([0, 1, 2])
    
    logger.info(f"Injected {noise_pct*100}% label noise into {num_noise} samples")

    # 5. Metadata and Time/Machine columns
    df["timestamp"] = pd.date_range(start="2024-01-01", periods=num_samples, freq="30s")
    df["machine_id"] = np.random.choice(["CNC-A01", "CNC-A02", "CNC-B01"], num_samples)

    # 6. Final Logging
    log_dataframe_summary(logger, df, "Raw Generated Data")
    
    sensor_stats = {}
    for feat in SENSOR_FEATURES:
        sensor_stats[feat] = f"Mean: {df[feat].mean():.2f}, Std: {df[feat].std():.2f}, Range: [{df[feat].min():.2f}, {df[feat].max():.2f}]"
    log_dict_table(logger, sensor_stats, "Sensor Statistics")

    # Save data
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW_DATA_PATH, index=False)
    logger.info(f"Raw data saved to {RAW_DATA_PATH} ({RAW_DATA_PATH.stat().st_size / 1024:.2f} KB)")

    # Save metadata
    metadata = {
        "generation_stats": {
            "total_samples": num_samples,
            "quality_distribution": df["quality"].value_counts().to_dict(),
            "machine_distribution": df["machine_id"].value_counts().to_dict(),
            "noise_percentage": noise_pct,
            "seed": 42
        }
    }
    save_metadata(metadata)

    if tracker and context_id is not None:
        raw_artifact_id = tracker.record_dataset(
            path=str(RAW_DATA_PATH), n_samples=len(df), version="raw-v1", context_id=context_id
        )
        tracker.record_execution(
            exec_type=tracker.generation_exec_type,
            properties={"n_samples": (len(df), "int"), "seed": (42, "int")},
            input_ids=[], output_ids=[raw_artifact_id], context_id=context_id
        )
        return str(RAW_DATA_PATH), raw_artifact_id
    return str(RAW_DATA_PATH), None

if __name__ == "__main__":
    run()
