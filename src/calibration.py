import numpy as np
from sklearn.calibration import CalibratedClassifierCV


def calibrate_model(model, X: np.ndarray, y: np.ndarray, method: str = "isotonic", cv: int = 3):
    """Aplica calibração isotônica para tornar probabilidades mais confiáveis."""
    if len(np.unique(y)) < 2:
        return model

    calibrated = CalibratedClassifierCV(model, cv=cv, method=method)
    return calibrated.fit(X, y)


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Calcula o Expected Calibration Error (ECE)."""
    if len(y_prob) == 0:
        return np.nan

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total = len(y_prob)

    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if i == n_bins - 1:
            mask = (y_prob >= bins[i]) & (y_prob <= bins[i + 1])

        if np.sum(mask) == 0:
            continue

        avg_prob = np.mean(y_prob[mask])
        avg_true = np.mean(y_true[mask])
        ece += (np.sum(mask) / total) * abs(avg_prob - avg_true)

    return float(ece)
