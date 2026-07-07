import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pypfopt.efficient_frontier import Efficient Frontier
from pypfopt import risk_models, expected_returns
import logging

def compute_historical_covariance(df_returns):
    """
    Calculates the clean sample covariance matrix of returns.
    """
    return risk_models.sample_cov(df_returns, returns_data=True)

def plot_covariance_heatmap(cov_matrix, save_path="data/processed/covariance_heatmap.png"):
    """
    Generates and saves a high-quality heatmap visualization of the covariance matrix.
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cov_matrix, annot=True, cmap="coolwarm", fmt=".6f", cbar=True)
    plt.title("Asset Covariance Matrix Heatmap", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def optimize_portfolio_mpt(expected_ret_vector, cov_matrix):
    """
    Extracts the Maximum Sharpe Ratio and Minimum Volatility portfolios.
    """
    logging.info("Generating Modern Portfolio Theory allocation weights...")
    
    # Max Sharpe Allocation
    ef_max_sharpe = EfficientFrontier(expected_ret_vector, cov_matrix)
    raw_weights_sharpe = ef_max_sharpe.max_sharpe(risk_free_rate=0.0)
    clean_weights_sharpe = ef_max_sharpe.clean_weights()
    perf_sharpe = ef_max_sharpe.portfolio_performance(verbose=False, risk_free_rate=0.0)
    
    # Min Volatility Allocation
    ef_min_vol = EfficientFrontier(expected_ret_vector, cov_matrix)
    raw_weights_vol = ef_min_vol.min_volatility()
    clean_weights_vol = ef_min_vol.clean_weights()
    perf_vol = ef_min_vol.portfolio_performance(verbose=False, risk_free_rate=0.0)
    
    return {
        "max_sharpe": {"weights": clean_weights_sharpe, "metrics": perf_sharpe},
        "min_vol": {"weights": clean_weights_vol, "metrics": perf_vol}
    }