"""Stacking ensemble and weighted voting for Mega Sena prediction."""

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import RobustScaler
from xgboost import XGBClassifier


def build_base_models(scale_pos_weight: float = 9.0) -> list[tuple[str, object]]:
    """Build the list of base estimators for the stacking ensemble.

    Args:
        scale_pos_weight: Ratio neg/pos for XGBoost imbalance handling.

    Returns:
        List of (name, estimator) tuples.
    """
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=3,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    gb = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        min_samples_split=5,
        min_samples_leaf=3,
        random_state=42,
    )

    xgb = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42,
        n_jobs=1,
        verbosity=0,
    )

    lr = LogisticRegression(
        max_iter=2000,
        C=0.5,
        penalty="l2",
        solver="lbfgs",
        class_weight="balanced",
        random_state=42,
    )

    return [("rf", rf), ("gb", gb), ("xgb", xgb), ("lr", lr)]


def build_stacking_classifier(scale_pos_weight: float = 9.0) -> StackingClassifier:
    """Build a stacking classifier with a logistic regression meta-learner.

    Args:
        scale_pos_weight: Ratio neg/pos for XGBoost.

    Returns:
        Configured StackingClassifier.
    """
    base_models = build_base_models(scale_pos_weight)

    meta_learner = LogisticRegression(
        max_iter=2000,
        C=1.0,
        class_weight="balanced",
        random_state=42,
    )

    return StackingClassifier(
        estimators=base_models,
        final_estimator=meta_learner,
        cv=3,
        stack_method="predict_proba",
        passthrough=False,
        n_jobs=1,
    )


def cv_weighted_ensemble(
    base_models: list[tuple[str, object]],
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_splits: int = 3,
) -> dict:
    """Train base models and compute CV-weighted ensemble weights.

    Each model's weight is proportional to its TimeSeriesSplit ROC-AUC.

    Args:
        base_models: List of (name, estimator) pairs.
        X_train: Training features.
        y_train: Training labels.
        n_splits: Number of CV folds.

    Returns:
        Dictionary with trained models and their normalized weights.
    """
    n_splits = min(n_splits, max(2, len(y_train) // 2))
    tscv = TimeSeriesSplit(n_splits=n_splits)

    cv_aucs: list[list[float]] = [[] for _ in base_models]

    for train_idx, val_idx in tscv.split(X_train):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]

        if len(np.unique(y_val)) < 2:
            continue

        for i, (_, model) in enumerate(base_models):
            try:
                model_clone = _clone_model(model)
                model_clone.fit(X_tr, y_tr)
                proba = model_clone.predict_proba(X_val)[:, 1]
                auc = roc_auc_score(y_val, proba)
                cv_aucs[i].append(auc)
            except Exception:
                cv_aucs[i].append(0.5)

    raw_weights = [
        float(np.mean(aucs)) if aucs else 0.5
        for aucs in cv_aucs
    ]

    raw_weights_arr = np.maximum(np.array(raw_weights) - 0.5, 0.0)
    total = float(np.sum(raw_weights_arr))
    if total > 0:
        normalized = (raw_weights_arr / total).tolist()
    else:
        normalized = [1.0 / len(base_models)] * len(base_models)

    trained_models = []
    for _, model in base_models:
        model.fit(X_train, y_train)
        trained_models.append(model)

    return {
        "models": trained_models,
        "weights": normalized,
        "cv_aucs": [float(np.mean(a)) if a else 0.5 for a in cv_aucs],
    }


def weighted_predict_proba(
    models: list,
    weights: list[float],
    X: np.ndarray,
) -> np.ndarray:
    """Weighted soft-voting prediction.

    Args:
        models: List of trained models.
        weights: Normalized weights for each model.
        X: Features to predict on.

    Returns:
        Weighted average probability array.
    """
    all_probas = []
    for model in models:
        proba = model.predict_proba(X)
        p = proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]
        all_probas.append(p)

    return np.average(np.array(all_probas), axis=0, weights=weights)


def _clone_model(model):
    """Create a fresh copy of a model with the same parameters."""
    from sklearn.base import clone
    return clone(model)
