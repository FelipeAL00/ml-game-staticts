"""Comprehensive metrics for imbalanced binary classification."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    matthews_corrcoef,
    average_precision_score,
    precision_recall_curve,
    auc,
)


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> dict:
    """Compute a comprehensive set of metrics for imbalanced classification.

    Args:
        y_true: True binary labels.
        y_pred: Predicted binary labels (after threshold).
        y_proba: Predicted probabilities for the positive class.

    Returns:
        Dictionary with all metrics.
    """
    metrics: dict[str, float] = {}

    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
    metrics["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))

    metrics["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    metrics["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
    metrics["f1"] = float(f1_score(y_true, y_pred, zero_division=0))

    try:
        metrics["mcc"] = float(matthews_corrcoef(y_true, y_pred))
    except ValueError:
        metrics["mcc"] = float("nan")

    if len(np.unique(y_true)) > 1:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        except ValueError:
            metrics["roc_auc"] = float("nan")

        try:
            metrics["pr_auc"] = float(average_precision_score(y_true, y_proba))
        except ValueError:
            metrics["pr_auc"] = float("nan")
    else:
        metrics["roc_auc"] = float("nan")
        metrics["pr_auc"] = float("nan")

    metrics["actual_prevalence"] = float(np.mean(y_true))
    metrics["predicted_prevalence"] = float(np.mean(y_pred))

    return metrics


def aggregate_metrics(metrics_list: list[dict]) -> dict:
    """Aggregate metrics across multiple classifiers (one per number).

    Args:
        metrics_list: List of metric dicts, one per number (1-60).

    Returns:
        Aggregated statistics.
    """
    keys = [k for k in metrics_list[0].keys() if k != "n"]
    agg: dict[str, dict] = {}

    for key in keys:
        values = [m[key] for m in metrics_list if not np.isnan(m.get(key, float("nan")))]
        if values:
            agg[key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "median": float(np.median(values)),
            }

    return agg


def print_aggregated_metrics(agg: dict) -> None:
    """Print aggregated metrics in a readable format."""
    priority_metrics = ["f1", "balanced_accuracy", "roc_auc", "pr_auc", "mcc", "precision", "recall", "accuracy"]

    print("\n   Metrica                     Media      Std       Min       Max")
    print("   " + "-" * 68)

    for key in priority_metrics:
        if key in agg:
            m = agg[key]
            print(
                f"   {key:<28s} {m['mean']:>8.4f}  {m['std']:>8.4f}  {m['min']:>8.4f}  {m['max']:>8.4f}"
            )
