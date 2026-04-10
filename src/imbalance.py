import warnings
import numpy as np
from sklearn.metrics import precision_recall_curve

try:
    from imblearn.over_sampling import SMOTE
except ModuleNotFoundError:  # pragma: no cover
    SMOTE = None


def balance_with_smote(X: np.ndarray, y: np.ndarray, random_state: int = 42):
    """Balanceia as classes usando SMOTE."""
    if len(np.unique(y)) < 2 or np.sum(y == 1) < 2:
        return X, y

    if SMOTE is None:
        warnings.warn(
            "imbalanced-learn não está instalado; o balanceamento com SMOTE será ignorado.",
            UserWarning,
        )
        return X, y

    smote = SMOTE(random_state=random_state)
    return smote.fit_resample(X, y)


def optimize_threshold_precision_recall(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Encontra o melhor limiar de decisão usando curva precisão-recall."""
    if len(np.unique(y_true)) < 2:
        return 0.5

    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-12)
    best_index = int(np.nanargmax(f1_scores))
    if best_index >= len(thresholds):
        return 0.5
    return float(thresholds[best_index])


def compute_scale_pos_weight(y: np.ndarray) -> float:
    """Computa o peso de classe para XGBoost em problemas desbalanceados."""
    neg = np.sum(y == 0)
    pos = np.sum(y == 1)
    if pos == 0:
        return 1.0
    return float(neg / pos)
