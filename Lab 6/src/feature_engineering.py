"""Stage 3: Feature engineering pipeline for CNC sensor data."""
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.utils import (
    get_logger, log_stage, log_dataframe_summary,
    LABELED_DATA_PATH, FEATURES_PATH,
    ALL_FEATURES, QUALITY_REVERSE_MAP,
    MODELS_DIR, FEATURE_SCALER_PATH,
    engineer_features_from_raw
)

logger = get_logger(__name__)


@log_stage("Feature Engineering")
def run():
    """Engineer features, normalize, encode target."""
    df = pd.read_csv(LABELED_DATA_PATH)
    logger.info(f"Loaded {len(df)} labeled records")

    # Engineer features using shared utility function
    df = engineer_features_from_raw(df)

    # Encode target
    df["quality_encoded"] = df["quality"].map(QUALITY_REVERSE_MAP)

    # Select feature columns and normalize
    feature_cols = ALL_FEATURES
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, FEATURE_SCALER_PATH)
    logger.info(f"Saved fitted scaler to {FEATURE_SCALER_PATH}")

    # Keep only what's needed for training
    output_cols = ["record_id"] + feature_cols + ["quality", "quality_encoded"]
    df_out = df[output_cols]

    FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(FEATURES_PATH, index=False)

    # Log feature stats
    for col in feature_cols:
        stats = df_out[col].describe()
        logger.info(f"  {col}: mean={stats['mean']:.3f}, std={stats['std']:.3f}, "
                     f"min={stats['min']:.3f}, max={stats['max']:.3f}")

    log_dataframe_summary(logger, df_out, "Engineered Features")
    logger.info(f"Saved features to {FEATURES_PATH}")
    return str(FEATURES_PATH)


if __name__ == "__main__":
    run()
