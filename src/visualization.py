"""
visualization.py

Visualization functions for exploratory data analysis (EDA)
and financial time series forecasting.

Author: Blen Debebe
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# ------------------------------------------------------------------
# Plot Settings
# ------------------------------------------------------------------

plt.style.use("ggplot")
sns.set_theme(style="whitegrid")

FIGURE_DIR = Path("figures")
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------
# Helper Function
# ------------------------------------------------------------------

def save_plot(filename: str):
    """
    Save the current figure.
    """

    plt.tight_layout()
    plt.savefig(FIGURE_DIR / filename, dpi=300)
    plt.close()


# ------------------------------------------------------------------
# Closing Price
# ------------------------------------------------------------------

def plot_closing_price(df: pd.DataFrame, asset: str):
    """
    Plot closing prices.
    """

    plt.figure(figsize=(14,6))

    plt.plot(
        df.index,
        df["Close"],
        linewidth=2,
    )

    plt.title(f"{asset} Closing Price")
    plt.xlabel("Date")
    plt.ylabel("Price ($)")

    save_plot(f"{asset.lower()}_closing_price.png")


# ------------------------------------------------------------------
# Daily Returns
# ------------------------------------------------------------------

def plot_daily_returns(df: pd.DataFrame, asset: str):
    """
    Plot percentage daily returns.
    """

    plt.figure(figsize=(14,6))

    plt.plot(
        df.index,
        df["Daily Return"],
    )

    plt.axhline(0, color="black", linestyle="--")

    plt.title(f"{asset} Daily Returns")
    plt.xlabel("Date")
    plt.ylabel("Return")

    save_plot(f"{asset.lower()}_daily_returns.png")


# ------------------------------------------------------------------
# Rolling Statistics
# ------------------------------------------------------------------

def plot_rolling_statistics(df: pd.DataFrame, asset: str):
    """
    Rolling mean and rolling standard deviation.
    """

    plt.figure(figsize=(14,6))

    plt.plot(
        df.index,
        df["Close"],
        label="Close",
    )

    plt.plot(
        df.index,
        df["Rolling Mean"],
        label="Rolling Mean",
    )

    plt.plot(
        df.index,
        df["Rolling Std"],
        label="Rolling Std",
    )

    plt.legend()

    plt.title(f"{asset} Rolling Statistics")

    save_plot(f"{asset.lower()}_rolling_statistics.png")


# ------------------------------------------------------------------
# Distribution
# ------------------------------------------------------------------

def plot_return_distribution(df: pd.DataFrame, asset: str):
    """
    Histogram of returns.
    """

    plt.figure(figsize=(10,6))

    sns.histplot(
        df["Daily Return"],
        kde=True,
        bins=40,
    )

    plt.title(f"{asset} Return Distribution")

    save_plot(f"{asset.lower()}_return_distribution.png")


# ------------------------------------------------------------------
# Boxplot
# ------------------------------------------------------------------

def plot_boxplot(df: pd.DataFrame, asset: str):
    """
    Detect outliers visually.
    """

    plt.figure(figsize=(8,6))

    sns.boxplot(
        y=df["Daily Return"]
    )

    plt.title(f"{asset} Daily Return Boxplot")

    save_plot(f"{asset.lower()}_boxplot.png")


# ------------------------------------------------------------------
# Volume
# ------------------------------------------------------------------

def plot_volume(df: pd.DataFrame, asset: str):
    """
    Trading volume.
    """

    plt.figure(figsize=(14,6))

    plt.plot(
        df.index,
        df["Volume"],
    )

    plt.title(f"{asset} Trading Volume")

    save_plot(f"{asset.lower()}_volume.png")


# ------------------------------------------------------------------
# Cumulative Returns
# ------------------------------------------------------------------

def plot_cumulative_returns(df: pd.DataFrame, asset: str):
    """
    Plot cumulative returns.
    """

    plt.figure(figsize=(14,6))

    plt.plot(
        df.index,
        df["Cumulative Return"],
    )

    plt.title(f"{asset} Cumulative Return")

    save_plot(f"{asset.lower()}_cumulative_return.png")


# ------------------------------------------------------------------
# Correlation Heatmap
# ------------------------------------------------------------------

def plot_correlation_heatmap(data: dict):
    """
    Correlation between assets.
    """

    combined = pd.DataFrame()

    for ticker, df in data.items():
        combined[ticker] = df["Close"]

    plt.figure(figsize=(8,6))

    sns.heatmap(
        combined.corr(),
        annot=True,
        cmap="coolwarm",
        linewidths=.5,
    )

    plt.title("Asset Correlation")

    save_plot("correlation_heatmap.png")


# ------------------------------------------------------------------
# Combined Closing Prices
# ------------------------------------------------------------------

def plot_all_assets(data: dict):
    """
    Compare closing prices.
    """

    plt.figure(figsize=(15,7))

    for ticker, df in data.items():

        plt.plot(
            df.index,
            df["Close"],
            label=ticker,
        )

    plt.legend()

    plt.title("Asset Closing Prices")

    plt.xlabel("Date")
    plt.ylabel("Price ($)")

    save_plot("all_assets_closing_price.png")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

if __name__ == "__main__":

    from src.data_loader import load_all_assets
    from src.preprocessing import prepare_dataset

    datasets = load_all_assets()

    processed = {}

    for ticker, df in datasets.items():

        processed_df = prepare_dataset(df)

        processed[ticker] = processed_df

        plot_closing_price(processed_df, ticker)
        plot_daily_returns(processed_df, ticker)
        plot_rolling_statistics(processed_df, ticker)
        plot_return_distribution(processed_df, ticker)
        plot_boxplot(processed_df, ticker)
        plot_volume(processed_df, ticker)
        plot_cumulative_returns(processed_df, ticker)

    plot_all_assets(processed)
    plot_correlation_heatmap(processed)

    print("All visualizations generated successfully!")