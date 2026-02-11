"""
Manufacturing Quality Prediction - High-Accuracy Model Training
Trains an optimized RandomForest classifier for CNC defect detection.
Includes detailed logging and hyperparameter tuning for pitch-perfect performance.
"""
import pandas as pd
import numpy as np
import json
import joblib
import os
import time
import logging
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, 
    accuracy_score, 
    confusion_matrix
)

# Configure logging for professional output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [MODEL_TRAINING] - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Full Terminology Feature Names
FEATURE_COLUMNS = [
    'spindle_speed',  # Revolutions Per Minute
    'feed_rate',      # Millimeters Per Minute
    'depth_of_cut',   # Millimeters
    'vibration',      # Millimeters Per Second
    'temperature',    # Degrees Celsius
    'tool_wear'       # Millimeters
]

def train_optimized_model(dataframe):
    """
    Train a highly accurate RandomForest model with tuned hyperparameters.
    """
    logger.info("Initializing model training pipeline...")
    
    X = dataframe[FEATURE_COLUMNS].values
    y = dataframe['quality_label'].values
    
    # Stratified split for balanced class representation
    logger.info("Splitting dataset (80% Train / 20% Test) with stratification...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Feature Scaling
    logger.info("Fitting and applying StandardScaler to sensor features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # High-Accuracy Hyperparameters
    logger.info("Configuring High-Accuracy RandomForest (200 Estimators, Depth 15)...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced_subsample' # Improved handling for synthetic data
    )
    
    logger.info("Starting model fitting (this may take a few moments)...")
    start_time = time.time()
    model.fit(X_train_scaled, y_train)
    duration = time.time() - start_time
    logger.info(f"Training completed successfully in {duration:.2f} seconds.")
    
    # Comprehensive Evaluation
    logger.info("Performing model evaluation on hold-out test set...")
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    target_names = ['Good Quality', 'Minor Defect', 'Major Defect']
    report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    logger.info(f"Final Model Accuracy: {accuracy:.2%}")
    
    # Extract Feature Importance (Full Terminology)
    importances = dict(zip(FEATURE_COLUMNS, model.feature_importances_.tolist()))
    
    metrics = {
        'model_name': 'Manufacturing Quality Classifier',
        'version': '2.0.0',
        'accuracy': float(accuracy),
        'hyperparameters': {
            'n_estimators': 200,
            'max_depth': 15,
            'class_weight': 'balanced_subsample'
        },
        'feature_importance_ranking': importances,
        'classification_report_full': report,
        'confusion_matrix': conf_matrix.tolist(),
        'training_timestamp': datetime.now().isoformat()
    }
    
    return model, scaler, metrics


if __name__ == '__main__':
    print("\n" + "="*80)
    print("  MODEL FACTORY: HIGH-ACCURACY TRAINING PIPELINE")
    print("="*80)
    
    data_source = '/exchange/manufacturing_data.csv'
    if not os.path.exists(data_source):
        logger.error(f"Data source not found at {data_source}. Aborting.")
        exit(1)
        
    logger.info(f"Loading manufacturing dataset from {data_source}...")
    df = pd.read_csv(data_source)
    
    # Execute Training
    final_model, final_scaler, final_metrics = train_optimized_model(df)
    
    # Save Artifacts
    output_path = '/exchange'
    os.makedirs(output_path, exist_ok=True)
    
    logger.info("Saving model and scaler artifacts...")
    joblib.dump(final_model, os.path.join(output_path, 'quality_model.joblib'))
    joblib.dump(final_scaler, os.path.join(output_path, 'scaler.joblib'))
    
    with open(os.path.join(output_path, 'training_metrics.json'), 'w') as f:
        json.dump(final_metrics, f, indent=4)
    
    logger.info(f"All artifacts saved to {output_path}")
    
    print("\n[PERFORMANCE SUMMARY]")
    print(f"  Accuracy Score: {final_metrics['accuracy']:.4f}")
    print("\n  Top Sensors by Importance:")
    sorted_importance = sorted(final_metrics['feature_importance_ranking'].items(), key=lambda x: x[1], reverse=True)
    for feat, imp in sorted_importance:
        print(f"    - {feat:20}: {imp:.4f}")
    
    print("\n" + "="*80)
    print("  TRAINING PIPELINE COMPLETE: READY FOR DEPLOYMENT")
    print("="*80 + "\n")
