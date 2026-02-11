"""
Manufacturing Quality Prediction - Synthetic Data Generator
Generates high-fidelity CNC machining sensor data for quality prediction.
Revised for full terminology and strict sensor limits.
"""
import numpy as np
import pandas as pd
import json
import os
import logging
from datetime import datetime

# Configure logging for detailed output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [DATA_GENERATOR] - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants for Sensor Limits (Full Terminology)
LIMITS = {
    'spindle_speed': {'min': 800.0, 'max': 5000.0, 'unit': 'Revolutions Per Minute'},
    'feed_rate': {'min': 50.0, 'max': 1000.0, 'unit': 'Millimeters Per Minute'},
    'depth_of_cut': {'min': 0.1, 'max': 10.0, 'unit': 'Millimeters'},
    'vibration': {'min': 0.1, 'max': 15.0, 'unit': 'Millimeters Per Second'},
    'temperature': {'min': 20.0, 'max': 500.0, 'unit': 'Degrees Celsius'},
    'tool_wear': {'min': 0.0, 'max': 1.0, 'unit': 'Millimeters'}
}

def generate_manufacturing_data(n_samples=5000, seed=42):
    """
    Generate synthetic manufacturing sensor data simulating precision CNC machining.
    
    Features (Full Terminology):
        - spindle_speed (Revolutions Per Minute)
        - feed_rate (Millimeters Per Minute)
        - depth_of_cut (Millimeters)
        - vibration (Millimeters Per Second)
        - temperature (Degrees Celsius)
        - tool_wear (Millimeters)
    """
    logger.info(f"Starting data generation for {n_samples} samples (Seed: {seed})...")
    np.random.seed(seed)
    
    # Generate base features within strict limits
    logger.info("Generating sensor feature distributions...")
    spindle_speed = np.random.uniform(LIMITS['spindle_speed']['min'], LIMITS['spindle_speed']['max'], n_samples)
    feed_rate = np.random.uniform(LIMITS['feed_rate']['min'], LIMITS['feed_rate']['max'], n_samples)
    depth_of_cut = np.random.uniform(LIMITS['depth_of_cut']['min'], LIMITS['depth_of_cut']['max'], n_samples)
    vibration = np.random.uniform(LIMITS['vibration']['min'], LIMITS['vibration']['max'], n_samples)
    temperature = np.random.uniform(LIMITS['temperature']['min'], LIMITS['temperature']['max'], n_samples)
    tool_wear = np.random.uniform(LIMITS['tool_wear']['min'], LIMITS['tool_wear']['max'], n_samples)

    # Create quality labels based on realistic manufacturing physics
    logger.info("Applying manufacturing physics rules for quality labeling...")
    labels = np.zeros(n_samples, dtype=int)
    
    for i in range(n_samples):
        defect_probability = 0.0
        
        # Vibration Rule
        if vibration[i] > 10.0: defect_probability += 0.8
        elif vibration[i] > 6.0: defect_probability += 0.4
            
        # Temperature Rule
        if temperature[i] > 400: defect_probability += 0.7
        elif temperature[i] > 300: defect_probability += 0.3
        
        # Tool Wear Rule
        if tool_wear[i] > 0.8: defect_probability += 0.9
        elif tool_wear[i] > 0.5: defect_probability += 0.4
        
        # Feed Rate / Spindle Speed Interaction (Chatter)
        if feed_rate[i] > 800 and spindle_speed[i] < 1500:
            defect_probability += 0.5
        
        # Add random process noise
        defect_score = defect_probability + np.random.normal(0, 0.1)
        
        if defect_score > 0.7:
            labels[i] = 2  # Major Defect
        elif defect_score > 0.35:
            labels[i] = 1  # Minor Defect
        else:
            labels[i] = 0  # Good Quality

    data = pd.DataFrame({
        'spindle_speed': np.round(spindle_speed, 1),
        'feed_rate': np.round(feed_rate, 1),
        'depth_of_cut': np.round(depth_of_cut, 2),
        'vibration': np.round(vibration, 2),
        'temperature': np.round(temperature, 1),
        'tool_wear': np.round(tool_wear, 3),
        'quality_label': labels
    })
    
    logger.info("Data generation and labeling complete.")
    return data


if __name__ == '__main__':
    print("\n" + "="*80)
    print("  MANUFACTURING ENGINE: HIGH-PRECISION DATA GENERATION")
    print("="*80)
    
    dataset = generate_manufacturing_data(n_samples=5000)
    
    output_directory = '/exchange'
    output_file = os.path.join(output_directory, 'manufacturing_data.csv')
    os.makedirs(output_directory, exist_ok=True)
    
    logger.info(f"Writing dataset to {output_file}...")
    dataset.to_csv(output_file, index=False)
    
    # Save comprehensive metadata
    metadata = {
        'project': 'Manufacturing Quality Prediction',
        'generated_at': datetime.now().isoformat(),
        'sensor_limits': LIMITS,
        'features': list(LIMITS.keys()),
        'target': 'quality_label',
        'classes': {0: 'Good Quality', 1: 'Minor Defect', 2: 'Major Defect'},
        'n_samples': len(dataset),
        'class_distribution': dataset['quality_label'].value_counts().to_dict()
    }
    
    metadata_file = os.path.join(output_directory, 'data_metadata.json')
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=4)
    
    logger.info(f"Metadata successfully saved to {metadata_file}")
    
    print("\n[SUMMARY REPORT]")
    print(f"  Total Samples: {len(dataset)}")
    print("  Label Distribution:")
    for label, count in dataset['quality_label'].value_counts().sort_index().items():
        name = metadata['classes'][label]
        percentage = (count / len(dataset)) * 100
        print(f"    - {name:15}: {count:5} ({percentage:.1f}%)")
    
    print("\n" + "="*80)
    print("  DATA GENERATION SUCCESSFUL")
    print("="*80 + "\n")
