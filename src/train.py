# train.py
# This script trains a machine learning model.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

def train_model():
    """
    Loads data, trains a machine learning model, and saves it.
    """
    # Load and preprocess data
    df = pd.read_csv("../data/processed/dataset.csv")  # Example file
    X = df.drop(columns=['target'])
    y = df['target']

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Save the trained model
    joblib.dump(model, "../models/model.pkl")

if __name__ == "__main__":
    train_model()
