import pandas as pd
import numpy as np

from src.preprocessing import (
    clean_data,
    calculate_daily_returns,
    calculate_rolling_statistics,
)


def sample_dataframe():
    dates = pd.date_range("2024-01-01", periods=10)

    data = {
        "Open": np.arange(10) + 100,
        "High": np.arange(10) + 101,
        "Low": np.arange(10) + 99,
        "Close": np.arange(10) + 100,
        "Adj Close": np.arange(10) + 100,
        "Volume": np.random.randint(1000, 5000, size=10),
    }

    return pd.DataFrame(data, index=dates)


def test_clean_data():

    df = sample_dataframe()

    df.iloc[3, 3] = np.nan

    cleaned = clean_data(df)

    assert cleaned.isnull().sum().sum() == 0


def test_daily_returns():

    df = sample_dataframe()

    returns = calculate_daily_returns(df)

    assert "Daily Return" in returns.columns


def test_rolling_statistics():

    df = sample_dataframe()

    rolling = calculate_rolling_statistics(df)

    assert "Rolling Mean" in rolling.columns

    assert "Rolling Std" in rolling.columns


def test_dataframe_not_empty():

    df = sample_dataframe()

    assert not df.empty


def test_close_column_exists():

    df = sample_dataframe()

    assert "Close" in df.columns