"""Stage 1: Generate synthetic CNC manufacturing sensor data."""
import argparse
import uuid
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from src.utils import (
    get_logger, log_stage, log_dataframe_summary,
    RAW_DATA_PATH, SENSOR_RANGES, QUALITY_CLASSES
)

logger = get_logger(__name__)


@log_stage("Data Generation")
def run(num_records=5000):
    """Generate realistic CNC sensor data with correlated defect labels."""
    np.random.seed(42)
    batch_id = str(uuid.uuid4())[:8]
    machine_ids = [f"CNC-{str(i).zfill(3)}" for i in range(1, 11)]
    base_time = datetime.now() - timedelta(days=30)

    records = []
    for i in range(num_records):
        # Generate sensor readings
        spindle_speed = np.random.uniform(1500, 4500)
        feed_rate = np.random.uniform(50, 400)
        depth_of_cut = np.random.uniform(0.5, 5.0)
        vibration = np.random.exponential(3.0) + 0.5
        vibration = np.clip(vibration, 0.5, 15.0)
        temperature = 150 + (spindle_speed / 4500) * 120 + np.random.normal(0, 20)
        temperature = np.clip(temperature, 150, 350)
        tool_wear = np.random.beta(2, 5)

        # Correlated quality label
        risk = (
            0.3 * (vibration / 15.0)
            + 0.25 * tool_wear
            + 0.15 * (temperature / 350.0)
            + 0.15 * (depth_of_cut / 5.0)
            + 0.15 * (feed_rate / 400.0)
        ) + np.random.normal(0, 0.05)

        if risk < 0.35:
            quality = "Good Quality"
        elif risk < 0.55:
            quality = "Minor Defect"
        else:
            quality = "Major Defect"

        records.append({
            "record_id": f"REC-{batch_id}-{str(i).zfill(5)}",
            "batch_id": batch_id,
            "machine_id": np.random.choice(machine_ids),
            "timestamp": (base_time + timedelta(minutes=i * 2)).isoformat(),
            "spindle_speed": round(spindle_speed, 2),
            "feed_rate": round(feed_rate, 2),
            "depth_of_cut": round(depth_of_cut, 3),
            "vibration": round(vibration, 3),
            "temperature": round(temperature, 2),
            "tool_wear": round(tool_wear, 4),
            "quality": quality,
        })

    df = pd.DataFrame(records)
    RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW_DATA_PATH, index=False)

    log_dataframe_summary(logger, df, "Raw Sensor Data")
    logger.info(f"Saved {len(df)} records to {RAW_DATA_PATH}")
    return str(RAW_DATA_PATH)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-records", type=int, default=5000)
    args = parser.parse_args()
    run(num_records=args.num_records)
