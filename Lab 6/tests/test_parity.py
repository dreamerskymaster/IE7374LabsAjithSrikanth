import pytest
import pandas as pd
import numpy as np
from src.utils import engineer_features_from_raw, SENSOR_FEATURES, ALL_FEATURES

def test_feature_engineering_parity():
    """Verify that feature engineering produces expected columns and non-null values."""
    # Create sample raw data
    data = {
        "spindle_speed": [2000, 3000],
        "feed_rate": [100, 200],
        "depth_of_cut": [1.0, 2.0],
        "vibration": [2.5, 5.0],
        "temperature": [200, 250],
        "tool_wear": [0.1, 0.4]
    }
    df = pd.DataFrame(data)
    
    # Apply transformation
    df_transformed = engineer_features_from_raw(df)
    
    # Check for new columns
    expected_new = [
        "vibration_temperature_ratio", 
        "tool_stress_index", 
        "spindle_efficiency", 
        "thermal_load", 
        "wear_rate"
    ]
    for col in expected_new:
        assert col in df_transformed.columns
        assert not df_transformed[col].isnull().any()

def test_feature_engineering_math():
    """Manually verify the math for one record."""
    data = {
        "spindle_speed": [2000.0],
        "feed_rate": [100.0],
        "depth_of_cut": [1.0],
        "vibration": [2.0],
        "temperature": [200.0],
        "tool_wear": [0.5]
    }
    df = pd.DataFrame(data)
    df_transformed = engineer_features_from_raw(df)
    
    row = df_transformed.iloc[0]
    
    # stress = wear * doc * feed = 0.5 * 1.0 * 100 = 50.0
    assert row["tool_stress_index"] == 50.0
    
    # thermal = temp * doc = 200 * 1.0 = 200.0
    assert row["thermal_load"] == 200.0
    
    # efficiency = speed / (vibration + 1.0) = 2000 / 3.0 = 666.666...
    assert pytest.approx(row["spindle_efficiency"]) == 2000.0 / 3.0

def test_feature_cols_constant():
    """Ensure ALL_FEATURES matches what we expect."""
    assert len(ALL_FEATURES) == 11
    assert all(f in ALL_FEATURES for f in SENSOR_FEATURES)
