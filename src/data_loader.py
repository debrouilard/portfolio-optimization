"""
data_loader.py

Handles downloading and loading financial market data using Yahoo Finance.
"""

from pathlib import Path
import logging

import pandas as pd
import yfinance as yf


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


RAW_DATA_DIR = Path("data/raw")
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)


def download_asset(
    ticker: str,
    start_date: str = "2015-01-01",
    end_date: str = "2026-06-30",
    save: bool = True,
) -> pd.DataFrame:
    """
    Download historical data for a single asset.

    Parameters
    ----------
    ticker : str
        Stock ticker.
    start_date : str
        Download start date.
    end_date : str
        Download end date.
    save : bool
        Whether to save CSV.

    Returns
    -------
    pd.DataFrame
    """

    try:

        logging.info(f"Downloading {ticker}...")

        df = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            auto_adjust=False,
            progress=False,
        )

        if df.empty:
            raise ValueError(f"No data downloaded for {ticker}")

        df.index.name = "Date"

        if save:
            filepath = RAW_DATA_DIR / f"{ticker}.csv"
            df.to_csv(filepath)
            logging.info(f"Saved to {filepath}")

        return df

    except Exception as e:
        logging.error(f"Failed downloading {ticker}: {e}")
        raise


def download_all_assets(
    tickers=None,
    start_date="2015-01-01",
    end_date="2026-06-30",
):
    """
    Download all project assets.

    Returns
    -------
    dict
        Dictionary of DataFrames.
    """

    if tickers is None:
        tickers = ["TSLA", "SPY", "BND"]

    datasets = {}

    for ticker in tickers:
        datasets[ticker] = download_asset(
            ticker,
            start_date,
            end_date,
        )

    return datasets


def load_asset(ticker: str) -> pd.DataFrame:
    """
    Load a saved CSV.

    Parameters
    ----------
    ticker : str

    Returns
    -------
    pd.DataFrame
    """

    filepath = RAW_DATA_DIR / f"{ticker}.csv"

    if not filepath.exists():
        raise FileNotFoundError(
            f"{filepath} not found. Run download_data.py first."
        )

    df = pd.read_csv(
        filepath,
        parse_dates=["Date"],
        index_col="Date",
    )

    return df


def load_all_assets(tickers=None):
    """
    Load all downloaded assets.

    Returns
    -------
    dict
    """

    if tickers is None:
        tickers = ["TSLA", "SPY", "BND"]

    datasets = {}

    for ticker in tickers:
        datasets[ticker] = load_asset(ticker)

    return datasets


def combine_assets(data_dict: dict) -> pd.DataFrame:
    """
    Combine multiple datasets.

    Returns
    -------
    pd.DataFrame
    """

    frames = []

    for ticker, df in data_dict.items():

        temp = df.copy()

        temp["Asset"] = ticker

        frames.append(temp)

    combined = pd.concat(frames)

    combined.sort_index(inplace=True)

    return combined


if __name__ == "__main__":

    assets = download_all_assets()

    combined = combine_assets(assets)

    print(combined.head())

    print("\nDownloaded Assets:")

    for name, data in assets.items():
        print(f"{name}: {data.shape}")