# test_train.py
# Unit tests for the training process.

import pytest
import pandas as pd
from src.train import train_model

def test_train_model():
    """
    Ensures that the train_model function runs without errors.
    """
    try:
        train_model()
    except Exception as e:
        pytest.fail(f"Training failed: {e}")
