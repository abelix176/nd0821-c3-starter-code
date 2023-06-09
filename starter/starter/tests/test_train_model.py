import pytest
import pandas as pd
import logging
from ml.data import process_data
from ml.model import train_model, inference, compute_model_metrics, evaluate_model_slices
import pickle
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import os

test_log = "logs/test.log"
logging.basicConfig(filename = test_log,
                    level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def test_train_model():
    # Dummy training data
    X_train = np.array([[1, 2, 3], [4, 5, 6]])
    y_train = np.array([0, 1])
    
    # Train the model
    model = train_model(X_train, y_train)
    
    try:
        # Assert that the model is trained (not None)
        assert model is not None
        logger.info("Test Model Training: SUCCESS")
    except AssertionError as err:
        logger.info("Test Model Training: FAILED")
        raise err

def test_compute_model_metrics():
    # Dummy labels and predictions
    y = np.array([0, 1, 1, 0])
    preds = np.array([0, 1, 0, 0])
    
    # Compute the model metrics
    precision, recall, fbeta = compute_model_metrics(y, preds)
    
    try:
        # Assert the computed metrics
        assert precision == 1.0
        assert recall == 0.5
        assert fbeta == 0.6666666666666666
        logger.info("Test Compute Model Metrics: SUCCESS")
    except AssertionError as err:
        logger.info("Test Compute Model Metrics: FAILED")
        raise err

def test_inference():
    # Dummy model and input data
    class DummyModel:
        def predict(self, X):
            return np.array([0, 1, 1])
    
    model = DummyModel()
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    
    # Perform inference
    predictions = inference(model, X)
    
    try:
        # Assert the predictions
        assert np.array_equal(predictions, np.array([0, 1, 1]))
        logger.info("Test Inference: SUCCESS")
    except AssertionError as err:
        logger.info("Test Inference: FAILED")
        raise err