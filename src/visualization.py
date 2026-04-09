"""Module for data visualization."""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def plot_feature_importance(model, feature_names: list[str], top_n: int = 10) -> None:
    """Plot feature importance from a tree-based model.

    Args:
        model: Trained model with feature_importances_ attribute.
        feature_names: List of feature names.
        top_n: Number of top features to display.
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[-top_n:]

    plt.figure(figsize=(10, 6))
    plt.barh(range(len(indices)), importances[indices])
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel("Importância")
    plt.title("Importância das Features")
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(cm: np.ndarray, labels: list[str] = None) -> None:
    """Plot a confusion matrix.

    Args:
        cm: Confusion matrix array.
        labels: Class labels.
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Previsto")
    plt.ylabel("Real")
    plt.title("Matriz de Confusão")
    plt.tight_layout()
    plt.show()


def plot_correlation_matrix(df: pd.DataFrame) -> None:
    """Plot correlation matrix of numeric features.

    Args:
        df: Input DataFrame.
    """
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()

    plt.figure(figsize=(12, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, fmt=".2f")
    plt.title("Matriz de Correlação")
    plt.tight_layout()
    plt.show()


def plot_distribution(df: pd.DataFrame, column: str, bins: int = 30) -> None:
    """Plot distribution of a numeric column.

    Args:
        df: Input DataFrame.
        column: Column name to plot.
        bins: Number of histogram bins.
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(df[column], bins=bins, kde=True)
    plt.xlabel(column)
    plt.ylabel("Frequência")
    plt.title(f"Distribuição: {column}")
    plt.tight_layout()
    plt.show()
