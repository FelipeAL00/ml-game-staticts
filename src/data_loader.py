"""Module for loading and preprocessing game statistics data."""

import pandas as pd
import numpy as np
from pathlib import Path


def load_csv(filepath: str) -> pd.DataFrame:
    """Load a CSV file into a pandas DataFrame."""
    return pd.read_csv(filepath)


def load_json(filepath: str) -> pd.DataFrame:
    """Load a JSON file into a pandas DataFrame."""
    return pd.read_json(filepath)


def get_data_path(filename: str, folder: str = "raw") -> Path:
    """Get the full path to a data file."""
    base_dir = Path(__file__).resolve().parent.parent / "data" / folder
    return base_dir / filename
