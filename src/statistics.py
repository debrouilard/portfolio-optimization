"""
statistics.py

Statistical analysis functions for financial time series.
"""

import numpy as np
import pandas as pd

from scipy.stats import zscore
from statsmodels.tsa.stattools import adfuller


# ------------------------------------------------------------------
# Basic Statistics
# ------------------------------------------------------------------

def descriptive_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate descriptive statistics.
    """

    stats = pd.DataFrame({
        "Mean": df.mean(numeric_only=True),
        "Median": df.median(numeric_only=True),
        "Std": df.std(numeric_only=True),
        "Variance": df.var(numeric_only=True),
        "Minimum": df.min(numeric_only=True),
        "Maximum": df.max(numeric_only=True),
        "Skewness": df.skew(numeric_only=True),
        "Kurtosis": df.kurtosis(numeric_only=True),
    })

    return stats


# ------------------------------------------------------------------
# Daily Returns
# ------------------------------------------------------------------

def calculate_daily_returns(df: pd.DataFrame) -> pd.Series:
    """
    Calculate percentage daily returns.
    """

    returns = df["Close"].pct_change()

    return returns.dropna()


# ------------------------------------------------------------------
# Volatility
# ------------------------------------------------------------------

def rolling_volatility(
    returns: pd.Series,
    window: int = 30,
) -> pd.Series:
    """
    Rolling standard deviation.
    """

    return returns.rolling(window).std()


# ------------------------------------------------------------------
# Outlier Detection
# ------------------------------------------------------------------

def detect_outliers_zscore(
    returns: pd.Series,
    threshold: float = 3,
):
    """
    Detect outliers using Z-score.
    """

    z_scores = np.abs(zscore(returns.dropna()))

    outliers = returns.dropna()[z_scores > threshold]

    return outliers


def detect_outliers_iqr(
    returns: pd.Series,
):
    """
    Detect outliers using IQR.
    """

    q1 = returns.quantile(0.25)
    q3 = returns.quantile(0.75)

    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    return returns[
        (returns < lower)
        | (returns > upper)
    ]


# ------------------------------------------------------------------
# Stationarity Test
# ------------------------------------------------------------------

def adf_test(series: pd.Series) -> dict:
    """
    Perform Augmented Dickey-Fuller Test.
    """

    series = series.dropna()

    result = adfuller(series)

    output = {
        "ADF Statistic": result[0],
        "p-value": result[1],
        "Lags Used": result[2],
        "Observations": result[3],
        "Critical Values": result[4],
    }

    return output


def print_adf_results(series, title="Series"):
    """
    Print formatted ADF results.
    """

    result = adf_test(series)

    print("=" * 60)
    print(f"ADF Test : {title}")
    print("=" * 60)

    print(f"ADF Statistic : {result['ADF Statistic']:.5f}")
    print(f"p-value       : {result['p-value']:.5f}")
    print(f"Lags Used     : {result['Lags Used']}")
    print(f"Observations  : {result['Observations']}")

    print("\nCritical Values")

    for key, value in result["Critical Values"].items():
        print(f"{key}: {value:.5f}")

    print()

    if result["p-value"] < 0.05:
        print("Result: Series is Stationary.")
    else:
        print("Result: Series is NOT Stationary.")


# ------------------------------------------------------------------
# Risk Metrics
# ------------------------------------------------------------------

def calculate_var(
    returns: pd.Series,
    confidence_level: float = 0.95,
):
    """
    Historical Value at Risk.
    """

    percentile = (1 - confidence_level) * 100

    return np.percentile(returns.dropna(), percentile)


def calculate_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.02,
):
    """
    Annualized Sharpe Ratio.
    """

    excess_return = returns - risk_free_rate / 252

    sharpe = (
        np.sqrt(252)
        * excess_return.mean()
        / excess_return.std()
    )

    return sharpe


def annualized_return(
    returns: pd.Series,
):
    """
    Annualized return.
    """

    return returns.mean() * 252


def annualized_volatility(
    returns: pd.Series,
):
    """
    Annualized volatility.
    """

    return returns.std() * np.sqrt(252)


# ------------------------------------------------------------------
# Complete Risk Report
# ------------------------------------------------------------------

def risk_report(df: pd.DataFrame):
    """
    Complete risk metrics report.
    """

    returns = calculate_daily_returns(df)

    report = {
        "Annual Return":
            annualized_return(returns),

        "Annual Volatility":
            annualized_volatility(returns),

        "Sharpe Ratio":
            calculate_sharpe_ratio(returns),

        "95% Value at Risk":
            calculate_var(returns),
    }

    return report


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

if __name__ == "__main__":

    from src.data_loader import load_asset

    df = load_asset("TSLA")

    returns = calculate_daily_returns(df)

    print(descriptive_statistics(df))

    print()

    print_adf_results(
        df["Close"],
        "Closing Price",
    )

    print()

    print_adf_results(
        returns,
        "Daily Returns",
    )

    print()

    print("Risk Report")

    print(risk_report(df))

    print()

    print(
        f"Number of Z-score Outliers: "
        f"{len(detect_outliers_zscore(returns))}"
    )

    print(
        f"Number of IQR Outliers: "
        f"{len(detect_outliers_iqr(returns))}"
    )