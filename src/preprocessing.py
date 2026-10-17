
from pathlib import Path
import json

import numpy as np
import pandas as pd

from src.feature_engineering import add_credit_features


# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent

FEATURES_PATH = BASE_DIR / "config" / "input_features.json"


def prepare_model_input(
    raw_data: pd.DataFrame,
    model_features: list
) -> pd.DataFrame:
    """
    Validate raw customer data, engineer features,
    and return the exact columns expected by
    the saved trained model.
    """

    if not isinstance(raw_data, pd.DataFrame):
        raise TypeError("Input must be a Pandas DataFrame.")

    # Load 23 required features
    with open(FEATURES_PATH, "r", encoding="utf-8") as f:
        required_features = json.load(f)

    # Check missing columns
    missing = [
        col for col in required_features
        if col not in raw_data.columns
    ]

    if missing:
        raise ValueError(f"Missing input features: {missing}")

    # Select raw features and convert to numeric
    df = raw_data[required_features].copy()

    for col in required_features:
        df[col] = pd.to_numeric(df[col], errors="raise")

    # Check missing and infinite values
    if df.isna().any().any():
        raise ValueError("Input contains missing values.")

    if not np.isfinite(df.to_numpy(dtype=float)).all():
        raise ValueError("Input contains infinite values.")

    # Prevent invalid utilization calculations
    if (df["LIMIT_BAL"] <= 0).any():
        raise ValueError("LIMIT_BAL must be greater than zero.")

    # Generate engineered features
    engineered_data = add_credit_features(df)

    # Select exact model features
    missing_model_features = [
        col for col in model_features
        if col not in engineered_data.columns
    ]

    if missing_model_features:
        raise ValueError(
            f"Missing model features: {missing_model_features}"
        )

    X = engineered_data.loc[:, model_features].copy()

    return X
