import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pypfopt.efficient_frontier import EfficientFrontier  # Fixed syntax spacing error
from pypfopt import risk_models
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def compute_historical_covariance(df_returns: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the clean sample covariance matrix of annualized returns.
    Ensures input data alignment and returns a structured pandas DataFrame.
    """
    if df_returns.empty:
        raise ValueError("Input returns DataFrame is empty.")
    # Annualizing the daily covariance matrix (252 trading days)
    return risk_models.sample_cov(df_returns, returns_data=True) * 252

def plot_covariance_heatmap(cov_matrix: pd.DataFrame, save_path: str = "data/processed/covariance_heatmap.png") -> None:
    """
    Generates and saves a high-quality heatmap visualization of the covariance matrix.
    """
    try:
        plt.figure(figsize=(8, 6))
        # Clear annotations with clean color balancing
        sns.heatmap(cov_matrix, annot=True, cmap="coolwarm", fmt=".6f", cbar=True, square=True)
        plt.title("Asset Covariance Matrix Heatmap", fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        logging.info(f"Covariance matrix heatmap successfully saved to {save_path}")
    except Exception as e:
        logging.error(f"Failed to generate or save covariance heatmap plot: {e}")
        raise e

def optimize_portfolio_mpt(expected_ret_vector: pd.Series, cov_matrix: pd.DataFrame) -> dict:
    """
    Extracts the Maximum Sharpe Ratio and Minimum Volatility portfolios using MPT.
    Includes explicit alignment checks and optimization error handling layers.
    """
    logging.info("Initializing Modern Portfolio Theory allocation weights routine...")
    
    # Absolute data alignment check between expectations and the covariance dimensions
    if not all(asset in cov_matrix.index for asset in expected_ret_vector.index):
        raise IndexError("Mismatch detected between Expected Returns index and Covariance Matrix dimensions.")

    try:
        # 1. Max Sharpe Allocation (Tangency Portfolio)
        ef_max_sharpe = EfficientFrontier(expected_ret_vector, cov_matrix)
        raw_weights_sharpe = ef_max_sharpe.max_sharpe(risk_free_rate=0.0)
        clean_weights_sharpe = ef_max_sharpe.clean_weights()
        perf_sharpe = ef_max_sharpe.portfolio_performance(verbose=False, risk_free_rate=0.0)
        
        # 2. Min Volatility Allocation
        ef_min_vol = EfficientFrontier(expected_ret_vector, cov_matrix)
        raw_weights_vol = ef_min_vol.min_volatility()
        clean_weights_vol = ef_min_vol.clean_weights()
        perf_vol = ef_min_vol.portfolio_performance(verbose=False, risk_free_rate=0.0)
        
        return {
            "max_sharpe": {
                "weights": dict(clean_weights_sharpe), 
                "metrics": {
                    "Expected Annual Return": perf_sharpe[0],
                    "Annual Volatility": perf_sharpe[1],
                    "Sharpe Ratio": perf_sharpe[2]
                }
            },
            "min_vol": {
                "weights": dict(clean_weights_vol), 
                "metrics": {
                    "Expected Annual Return": perf_vol[0],
                    "Annual Volatility": perf_vol[1],
                    "Sharpe Ratio": perf_vol[2]
                }
            }
        }
    except Exception as e:
        logging.error(f"Portfolio optimization failure encountered during algorithmic convergence: {e}")
        raise e