# evaluate.py
# This script evaluates a trained model using test data.

import pandas as pd
import joblib
from sklearn.metrics import accuracy_score

def evaluate_model():
    """
    Loads a trained model and evaluates its performance on test data.
    """
    # Load test data
    df = pd.read_csv("../data/processed/dataset.csv")  # Example file
    X_test = df.drop(columns=['target'])
    y_test = df['target']

    # Load trained model
    model = joblib.load("../models/model.pkl")

    # Make predictions
    y_pred = model.predict(X_test)

    # Compute accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.2f}")

if __name__ == "__main__":
    evaluate_model()
