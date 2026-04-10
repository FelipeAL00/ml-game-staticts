import numpy as np
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
    balanced_accuracy_score,
    precision_score,
    recall_score,
)


def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    """Calcula métricas adequadas para problemas desbalanceados."""
    metrics = {
        "f1": np.nan,
        "precision": np.nan,
        "recall": np.nan,
        "pr_auc": np.nan,
        "roc_auc": np.nan,
        "mcc": np.nan,
        "balanced_accuracy": np.nan,
    }

    if len(np.unique(y_true)) >= 2:
        try:
            metrics["f1"] = f1_score(y_true, y_pred, zero_division=0)
        except ValueError:
            metrics["f1"] = np.nan

        try:
            metrics["precision"] = precision_score(y_true, y_pred, zero_division=0)
        except ValueError:
            metrics["precision"] = np.nan

        try:
            metrics["recall"] = recall_score(y_true, y_pred, zero_division=0)
        except ValueError:
            metrics["recall"] = np.nan

        try:
            metrics["pr_auc"] = average_precision_score(y_true, y_prob)
        except ValueError:
            metrics["pr_auc"] = np.nan

        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
        except ValueError:
            metrics["roc_auc"] = np.nan

        try:
            metrics["mcc"] = matthews_corrcoef(y_true, y_pred)
        except ValueError:
            metrics["mcc"] = np.nan

        try:
            metrics["balanced_accuracy"] = balanced_accuracy_score(y_true, y_pred)
        except ValueError:
            metrics["balanced_accuracy"] = np.nan

    return metrics
