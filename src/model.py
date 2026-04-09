"""Module for training and evaluating ML models."""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
from pathlib import Path


MODELS = {
    "random_forest": RandomForestClassifier,
    "gradient_boosting": GradientBoostingClassifier,
    "logistic_regression": LogisticRegression,
}


def split_data(
    df: pd.DataFrame,
    target_column: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple:
    """Split data into train and test sets.

    Args:
        df: Input DataFrame.
        target_column: Name of the target column.
        test_size: Proportion of data for testing.
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test).
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_name: str = "random_forest",
    **kwargs,
):
    """Train a machine learning model.

    Args:
        X_train: Training features.
        y_train: Training labels.
        model_name: Name of the model to train.
        **kwargs: Additional parameters for the model.

    Returns:
        Trained model.
    """
    if model_name not in MODELS:
        raise ValueError(f"Unknown model: {model_name}. Choose from: {list(MODELS.keys())}")

    model = MODELS[model_name](**kwargs)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Evaluate a trained model.

    Args:
        model: Trained model.
        X_test: Test features.
        y_test: Test labels.

    Returns:
        Dictionary with evaluation metrics.
    """
    y_pred = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
    }


def cross_validate(model, X: np.ndarray, y: np.ndarray, cv: int = 5) -> dict:
    """Perform cross-validation.

    Args:
        model: Model to evaluate.
        X: Features.
        y: Labels.
        cv: Number of cross-validation folds.

    Returns:
        Dictionary with CV scores.
    """
    scores = cross_val_score(model, X, y, cv=cv)
    return {
        "mean_score": scores.mean(),
        "std_score": scores.std(),
        "scores": scores,
    }


def save_model(model, filename: str) -> Path:
    """Save a trained model to disk.

    Args:
        model: Trained model.
        filename: Name of the file.

    Returns:
        Path to the saved model.
    """
    models_dir = Path(__file__).resolve().parent.parent / "models"
    models_dir.mkdir(exist_ok=True)
    filepath = models_dir / filename
    joblib.dump(model, filepath)
    return filepath


def load_model(filename: str):
    """Load a trained model from disk.

    Args:
        filename: Name of the model file.

    Returns:
        Loaded model.
    """
    models_dir = Path(__file__).resolve().parent.parent / "models"
    filepath = models_dir / filename
    return joblib.load(filepath)
