"""Module for data preprocessing and feature engineering."""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder


def handle_missing_values(df: pd.DataFrame, strategy: str = "mean") -> pd.DataFrame:
    """Handle missing values in the DataFrame.

    Args:
        df: Input DataFrame.
        strategy: Strategy for filling missing values ('mean', 'median', 'drop').

    Returns:
        DataFrame with missing values handled.
    """
    if strategy == "mean":
        return df.fillna(df.select_dtypes(include=[np.number]).mean())
    elif strategy == "median":
        return df.fillna(df.select_dtypes(include=[np.number]).median())
    elif strategy == "drop":
        return df.dropna()
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def scale_features(df: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, StandardScaler]:
    """Scale numeric features using StandardScaler.

    Args:
        df: Input DataFrame.
        columns: List of column names to scale.

    Returns:
        Tuple of (scaled DataFrame, fitted scaler).
    """
    scaler = StandardScaler()
    df[columns] = scaler.fit_transform(df[columns])
    return df, scaler


def encode_labels(df: pd.DataFrame, column: str) -> tuple[pd.DataFrame, LabelEncoder]:
    """Encode categorical labels.

    Args:
        df: Input DataFrame.
        column: Column name to encode.

    Returns:
        Tuple of (encoded DataFrame, fitted encoder).
    """
    encoder = LabelEncoder()
    df[column] = encoder.fit_transform(df[column])
    return df, encoder
