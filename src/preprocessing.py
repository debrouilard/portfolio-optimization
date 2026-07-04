"""
preprocessing.py

Functions for cleaning and preprocessing financial time series data.
"""

from pathlib import Path
import logging

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

PROCESSED_DATA_DIR = Path("data/processed")
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean financial dataset.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """

    df = df.copy()

    # Remove duplicate rows
    df = df[~df.index.duplicated(keep="first")]

    # Convert columns to numeric
    numeric_columns = [
        "Open",
        "High",
        "Low",
        "Close",
        "Adj Close",
        "Volume",
    ]

    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    # Handle missing values
    df = df.ffill()
    df = df.bfill()
    df = df.interpolate(method="linear")

    # Remove remaining missing rows
    df.dropna(inplace=True)

    return df


def check_missing_values(df: pd.DataFrame) -> pd.Series:
    """
    Count missing values.
    """

    return df.isnull().sum()


def data_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summary statistics.
    """

    return df.describe().T


def calculate_daily_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate percentage daily returns.
    """

    df = df.copy()

    df["Daily Return"] = df["Close"].pct_change()

    df.dropna(inplace=True)

    return df


def calculate_log_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate log returns.
    """

    df = df.copy()

    df["Log Return"] = np.log(df["Close"] / df["Close"].shift(1))

    df.dropna(inplace=True)

    return df


def calculate_cumulative_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate cumulative returns.
    """

    df = calculate_daily_returns(df)

    df["Cumulative Return"] = (
        1 + df["Daily Return"]
    ).cumprod()

    return df


def calculate_rolling_statistics(
    df: pd.DataFrame,
    window: int = 30,
) -> pd.DataFrame:
    """
    Rolling statistics.
    """

    df = df.copy()

    df["Rolling Mean"] = (
        df["Close"]
        .rolling(window=window)
        .mean()
    )

    df["Rolling Std"] = (
        df["Close"]
        .rolling(window=window)
        .std()
    )

    return df


def normalize_close_price(df: pd.DataFrame) -> pd.DataFrame:
    """
    Min-Max normalization.
    """

    df = df.copy()

    minimum = df["Close"].min()
    maximum = df["Close"].max()

    df["Normalized Close"] = (
        df["Close"] - minimum
    ) / (maximum - minimum)

    return df


def standardize_close_price(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize closing prices.
    """

    df = df.copy()

    mean = df["Close"].mean()
    std = df["Close"].std()

    df["Standardized Close"] = (
        df["Close"] - mean
    ) / std

    return df


def prepare_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Complete preprocessing pipeline.
    """

    logging.info("Cleaning data...")

    df = clean_data(df)

    logging.info("Calculating daily returns...")

    df = calculate_daily_returns(df)

    logging.info("Calculating rolling statistics...")

    df = calculate_rolling_statistics(df)

    logging.info("Calculating cumulative returns...")

    df = calculate_cumulative_returns(df)

    logging.info("Normalizing prices...")

    df = normalize_close_price(df)

    logging.info("Standardizing prices...")

    df = standardize_close_price(df)

    return df


def save_processed_data(
    df: pd.DataFrame,
    filename: str,
) -> None:
    """
    Save processed dataset.
    """

    filepath = PROCESSED_DATA_DIR / filename

    df.to_csv(filepath)

    logging.info(f"Saved processed dataset to {filepath}")


def load_processed_data(
    filename: str,
) -> pd.DataFrame:
    """
    Load processed dataset.
    """

    filepath = PROCESSED_DATA_DIR / filename

    if not filepath.exists():
        raise FileNotFoundError(
            f"{filepath} does not exist."
        )

    return pd.read_csv(
        filepath,
        parse_dates=["Date"],
        index_col="Date",
    )


if __name__ == "__main__":

    from src.data_loader import load_asset

    df = load_asset("TSLA")

    processed = prepare_dataset(df)

    save_processed_data(
        processed,
        "TSLA_processed.csv",
    )

    print(processed.head())

    print("\nMissing Values")

    print(check_missing_values(processed))

    print("\nSummary Statistics")

    print(data_summary(processed))