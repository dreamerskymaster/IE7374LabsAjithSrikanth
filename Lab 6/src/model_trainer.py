"""Stage 4: Train RandomForest classifier and save model with metadata."""
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
from src.utils import (
    get_logger, log_stage, log_dict_table,
    FEATURES_PATH, MODEL_PATH, MODEL_META_PATH, METRICS_PATH,
    ALL_FEATURES, QUALITY_MAP, PIPELINE_VERSION, MODELS_DIR
)

logger = get_logger(__name__)


@log_stage("Model Training")
def run():
    """Train RandomForest on engineered features, save model and metadata."""
    df = pd.read_csv(FEATURES_PATH)
    logger.info(f"Loaded {len(df)} feature records")

    X = df[ALL_FEATURES]
    y = df["quality_encoded"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    logger.info(f"Train: {len(X_train)}, Test: {len(X_test)}")

    # Train
    model = RandomForestClassifier(
        n_estimators=200, max_depth=15, min_samples_split=5,
        class_weight="balanced", random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_test, y_pred).tolist()

    metrics = {
        "accuracy": round(acc, 4),
        "precision_macro": round(prec, 4),
        "recall_macro": round(rec, 4),
        "f1_macro": round(f1, 4),
        "confusion_matrix": cm,
    }
    log_dict_table(logger, {k: v for k, v in metrics.items() if k != "confusion_matrix"}, "Test Metrics")
    logger.info(f"Confusion Matrix:\n{np.array(cm)}")
    logger.info(f"\n{classification_report(y_test, y_pred, target_names=list(QUALITY_MAP.values()))}")

    # Feature importance
    importances = dict(zip(ALL_FEATURES, [round(float(x), 4) for x in model.feature_importances_]))
    log_dict_table(logger, dict(sorted(importances.items(), key=lambda x: x[1], reverse=True)[:5]), "Top 5 Features")

    # Save model
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    logger.info(f"Model saved to {MODEL_PATH}")

    # Save metadata
    meta = {
        "model_type": "RandomForest",
        "n_estimators": 200,
        "max_depth": 15,
        "accuracy": metrics["accuracy"],
        "f1_macro": metrics["f1_macro"],
        "training_samples": len(X_train),
        "test_samples": len(X_test),
        "feature_columns": ALL_FEATURES,
        "feature_importances": importances,
        "trained_at": datetime.now().isoformat(),
        "pipeline_version": PIPELINE_VERSION,
    }
    with open(MODEL_META_PATH, "w") as f:
        json.dump(meta, f, indent=2)

    # Save DVC metrics file
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Metadata saved to {MODEL_META_PATH}")
    logger.info(f"Metrics saved to {METRICS_PATH}")
    return str(MODEL_PATH)


if __name__ == "__main__":
    run()
