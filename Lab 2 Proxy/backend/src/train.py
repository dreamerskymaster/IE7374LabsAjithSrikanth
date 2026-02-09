import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

def train_model():
    """
    Trains a Random Forest Classifier on the Iris dataset.
    
    This script performs the following:
    1. Loads the Iris dataset from sklearn.
    2. Splits the data into training and testing sets.
    3. Trains a RandomForestClassifier.
    4. Saves the trained model to a pickle file for deployment.
    
    Author: Ajith Srikanth (IE7374 - MLOps)
    """
    print("--- Starting Model Training Phase ---")
    
    # Load the classic Iris dataset
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = iris.target
    
    # Split data: 80% training, 20% testing
    # Using a fixed random state ensures consistency for this lab environment
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Data split successfully. Training samples: {len(X_train)}")
    
    # Initialize and train our classifier
    # We use Random Forest for its robustness and ease of interpretation
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    print("Model training completed successfully.")
    
    # Ensure the model directory exists
    model_dir = "Lab 2/backend/model"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        
    model_path = os.path.join(model_dir, "iris_model.pkl")
    
    # Serialize the model to disk
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
        
    print(f"Model saved efficiently at: {model_path}")
    print("--- Training Phase Finalized ---")

if __name__ == "__main__":
    train_model()
