"""Utilities for handling class imbalance in binary classification."""

import numpy as np
from sklearn.metrics import roc_curve, f1_score, precision_recall_curve, auc


def find_optimal_threshold(y_true: np.ndarray, y_proba: np.ndarray, method: str = "f1") -> float:
    """Find the optimal decision threshold for a binary classifier.

    Args:
        y_true: True binary labels.
        y_proba: Predicted probabilities for the positive class.
        method: Optimization criterion ('f1', 'youden', 'precision_recall').

    Returns:
        Optimal threshold value.
    """
    if method == "youden":
        fpr, tpr, thresholds = roc_curve(y_true, y_proba)
        idx = int(np.argmax(tpr - fpr))
        return float(thresholds[idx])

    if method == "precision_recall":
        precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
        f1_scores = np.where(
            (precision + recall) > 0,
            2 * precision * recall / (precision + recall),
            0.0,
        )
        idx = int(np.argmax(f1_scores))
        return float(thresholds[idx]) if idx < len(thresholds) else 0.5

    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    f1_scores = np.where(
        (precision + recall) > 0,
        2 * precision * recall / (precision + recall),
        0.0,
    )
    idx = int(np.argmax(f1_scores[:-1]))
    return float(thresholds[idx]) if len(thresholds) > 0 else 0.5


def apply_threshold(y_proba: np.ndarray, threshold: float) -> np.ndarray:
    """Apply a decision threshold to predicted probabilities."""
    return (y_proba >= threshold).astype(int)


def smote_resample(
    X: np.ndarray,
    y: np.ndarray,
    sampling_strategy: float = 0.3,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Resample training data using SMOTE to address class imbalance.

    Falls back to class_weight='balanced' approach if imbalanced-learn
    is not installed.

    Args:
        X: Training features.
        y: Training labels.
        sampling_strategy: Target ratio of minority to majority class after resampling.
        random_state: Random seed.

    Returns:
        Tuple of (resampled X, resampled y).
    """
    try:
        from imblearn.over_sampling import SMOTE
        from imblearn.under_sampling import RandomUnderSampler

        n_pos = int(np.sum(y))
        n_neg = len(y) - n_pos

        if n_pos < 5:
            return X, y

        smote = SMOTE(
            sampling_strategy=sampling_strategy,
            k_neighbors=min(5, n_pos - 1),
            random_state=random_state,
        )
        under = RandomUnderSampler(
            sampling_strategy=min(0.6, n_pos / n_neg * 6),
            random_state=random_state,
        )

        X_res, y_res = smote.fit_resample(X, y)
        X_res, y_res = under.fit_resample(X_res, y_res)
        return X_res, y_res

    except ImportError:
        return X, y


def compute_scale_pos_weight(y: np.ndarray) -> float:
    """Compute scale_pos_weight for XGBoost from class distribution."""
    n_neg = float(np.sum(y == 0))
    n_pos = float(np.sum(y == 1))
    if n_pos == 0:
        return 1.0
    return n_neg / n_pos
