"""Functions to load and preprocess data."""

import pandas as pd
from src.config import DATA_RAW_PATH

def load_data(filename):
    """Load dataset from the raw data folder."""
    return pd.read_csv(f"{DATA_RAW_PATH}{filename}")
