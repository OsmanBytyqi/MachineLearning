# inference.py
# This script loads a trained model and makes predictions on new data.

import joblib
import pandas as pd

def make_predictions(new_data):
    """
    Makes predictions using a trained model.

    Args:
        new_data (pd.DataFrame): Data for prediction.

    Returns:
        list: Predicted values.
    """
    # Load the trained model
    model = joblib.load("../models/model.pkl")

    # Predict outcomes
    predictions = model.predict(new_data)
    return predictions

if __name__ == "__main__":
    # Example input
    example_data = pd.DataFrame([[5.1, 3.5, 1.4, 0.2]], columns=['feature1', 'feature2', 'feature3', 'feature4'])
    preds = make_predictions(example_data)
    print("Predictions:", preds)
