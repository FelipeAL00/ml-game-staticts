"""Probability calibration utilities for better reliability estimates."""

import numpy as np
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression


def calibrate_model(model, X_val: np.ndarray, y_val: np.ndarray, method: str = "isotonic"):
    """Calibrate a trained model's probabilities using a held-out validation set.

    Args:
        model: Trained sklearn-compatible classifier.
        X_val: Validation features.
        y_val: Validation labels.
        method: Calibration method ('isotonic' or 'sigmoid').

    Returns:
        Calibrated classifier.
    """
    calibrated = CalibratedClassifierCV(
        estimator=model,
        method=method,
        cv="prefit",
    )
    calibrated.fit(X_val, y_val)
    return calibrated


def expected_calibration_error(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error (ECE).

    Lower ECE means better-calibrated probabilities.

    Args:
        y_true: True binary labels.
        y_proba: Predicted probabilities.
        n_bins: Number of calibration bins.

    Returns:
        ECE value (0 = perfect, 1 = worst).
    """
    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=n_bins, strategy="uniform")
    ece = float(np.mean(np.abs(prob_true - prob_pred)))
    return ece


def reliability_summary(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10,
) -> dict:
    """Compute calibration reliability summary.

    Args:
        y_true: True binary labels.
        y_proba: Predicted probabilities.
        n_bins: Number of calibration bins.

    Returns:
        Dictionary with calibration stats.
    """
    try:
        prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=n_bins, strategy="uniform")
        ece = float(np.mean(np.abs(prob_true - prob_pred)))
        max_error = float(np.max(np.abs(prob_true - prob_pred)))
        overconf = float(np.mean(prob_pred - prob_true))
    except ValueError:
        ece = float("nan")
        max_error = float("nan")
        overconf = float("nan")

    return {
        "ece": ece,
        "max_calibration_error": max_error,
        "overconfidence": overconf,
    }
