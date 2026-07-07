import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def run_portfolio_backtest(df_returns, strategy_weights, benchmark_weights=None):
    """
    Simulates cumulative performance metrics over a specific backtesting window.
    Default Benchmark: Static Balanced Asset Allocation Strategy (60% SPY / 40% BND)
    """
    if benchmark_weights is None:
        benchmark_weights = {"TSLA": 0.0, "SPY": 0.6, "BND": 0.4}
        
    assets = list(strategy_weights.keys())
    
    # Align structural column layouts
    strat_w_vector = np.array([strategy_weights[asset] for asset in assets])
    bench_w_vector = np.array([benchmark_weights.get(asset, 0.0) for asset in assets])
    
    # Performance stream calculations
    strat_daily_returns = df_returns[assets].dot(strat_w_vector)
    bench_daily_returns = df_returns[assets].dot(bench_w_vector)
    
    strat_cum = (1 + strat_daily_returns).cumprod() - 1
    bench_cum = (1 + bench_daily_returns).cumprod() - 1
    
    return strat_daily_returns, bench_daily_returns, strat_cum, bench_cum

def calculate_backtest_metrics(daily_returns_series):
    """
    Extracts total return, annualized return, Sharpe Ratio, and Maximum Drawdown.
    """
    total_return = (1 + daily_returns_series).prod() - 1
    n_days = len(daily_returns_series)
    annualized_return = (1 + total_return) ** (252 / n_days) - 1
    
    vol = daily_returns_series.std() * np.sqrt(252)
    sharpe = (daily_returns_series.mean() / daily_returns_series.std()) * np.sqrt(252) if vol != 0 else 0
    
    # Maximum Drawdown calculation
    cum_wealth = (1 + daily_returns_series).cumprod()
    running_max = cum_wealth.cummax()
    drawdown = (cum_wealth - running_max) / running_max
    max_drawdown = drawdown.min()
    
    return {
        "Total Return": total_return,
        "Annualized Return": annualized_return,
        "Sharpe Ratio": sharpe,
        "Max Drawdown": max_drawdown
    }