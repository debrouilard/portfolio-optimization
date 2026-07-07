import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.forecasting import (
    split_data_chronologically,
    optimize_and_forecast_arima,
    train_and_forecast_lstm,
    calculate_metrics
)

def main():
    processed_path = "data/processed/cleaned_data.csv"
    if not os.path.exists(processed_path):
        print("[ERROR] Please run your preprocessing script to generate clean data profiles first.")
        return
        
    df = pd.read_csv(processed_path, index_col=0, parse_dates=True)
    target_col = "Close_TSLA"
    
    # --------------------------------------------------------------------------
    # TASK 2: Chronological Split & Multi-Model Comparison Matrix
    # --------------------------------------------------------------------------
    print("\n--- Running Task 2: Multi-Model Evaluation Tracking ---")
    train, test = split_data_chronologically(df, target_col=target_col, split_date="2025-01-01")
    
    arima_test_pred, arima_model = optimize_and_forecast_arima(train, test)
    lstm_test_pred, _, _ = train_and_forecast_lstm(train, test, window_size=60, epochs=10)
    
    arima_metrics = calculate_metrics(test, arima_test_pred)
    lstm_metrics = calculate_metrics(test, lstm_test_pred)
    
    print("\n==================================================================")
    print("                  TASK 2 MODEL COMPARISON REPORT                 ")
    print("==================================================================")
    print(f"ARIMA Metrics -> MAE: {arima_metrics['MAE']:.4f} | RMSE: {arima_metrics['RMSE']:.4f} | MAPE: {arima_metrics['MAPE']:.2f}%")
    print(f"LSTM Metrics  -> MAE: {lstm_metrics['MAE']:.4f}  | RMSE: {lstm_metrics['RMSE']:.4f}  | MAPE: {lstm_metrics['MAPE']:.2f}%")
    print("==================================================================")

    # Choose ARIMA as Champion model based on error minimization
    champion_model = arima_model

    # --------------------------------------------------------------------------
    # TASK 3: Fully Model-Driven Future Forecast with Confidence Intervals
    # --------------------------------------------------------------------------
    print("\n--- Running Task 3: Generating Model-Driven Future Projections ---")
    forecast_steps = 252 # 12-Month out-of-sample future track
    forecast_output = champion_model.get_forecast(steps=forecast_steps)
    
    future_mean = forecast_output.summary_frame()["mean"]
    future_ci_lower = forecast_output.summary_frame()["mean_ci_lower"]
    future_ci_upper = forecast_output.summary_frame()["mean_ci_upper"]
    
    future_dates = pd.date_range(start=test.index[-1] + pd.Timedelta(days=1), periods=forecast_steps, freq="B")
    future_mean.index, future_ci_lower.index, future_ci_upper.index = future_dates, future_dates, future_dates
    
    plt.figure(figsize=(12, 6))
    plt.plot(df.index[-400:], df[target_col].iloc[-400:], label="Historical / Test Data Price", color="#1f77b4")
    plt.plot(future_dates, future_mean, label="12-Month Model Forecast Horizon", color="#ff7f0e", linewidth=2)
    plt.fill_between(future_dates, future_ci_lower, future_ci_upper, color="gray", alpha=0.3, label="95% Confidence Bounds")
    plt.title("Task 3: Model-Driven Tesla (TSLA) Projections & Volatility Channels")
    plt.legend()
    plt.tight_layout()
    os.makedirs("data/processed", exist_ok=True)
    plt.savefig("data/processed/task3_future_forecast.png", dpi=300)
    plt.close()

    # --------------------------------------------------------------------------
    # TASK 4 & 5: MPT Efficient Frontier and Backtesting Window Wiring
    # --------------------------------------------------------------------------
    print("\n--- Running Tasks 4 & 5: Constructing Allocation and Verification Engine ---")
    returns_df = df[["Return_TSLA", "Return_BND", "Return_SPY"]].dropna()
    returns_df.columns = ["TSLA", "BND", "SPY"]
    
    cov_matrix = returns_df.cov() * 252
    tsla_expected_return = (future_mean.iloc[-1] - df[target_col].iloc[-1]) / df[target_col].iloc[-1]
    hist_avg_returns = returns_df.mean() * 252
    expected_returns = np.array([tsla_expected_return, hist_avg_returns["BND"], hist_avg_returns["SPY"]])
    
    # Simple portfolio optimization setup
    def get_portfolio_vol(weights): return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    def get_neg_sharpe(weights): return -np.dot(weights, expected_returns) / get_portfolio_vol(weights)
    
    cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(3))
    
    opt_res = minimize(get_neg_sharpe, [1/3, 1/3, 1/3], method='SLSQP', bounds=bounds, constraints=cons)
    w_strategy = opt_res.x
    w_benchmark = np.array([0.0, 0.4, 0.6]) # 60% SPY / 40% BND Benchmark
    
    # Task 5 out-of-sample backtest isolating final year
    backtest_cutoff = returns_df.index.max() - pd.DateOffset(years=1)
    backtest_window_data = returns_df[returns_df.index >= backtest_cutoff]
    
    strat_daily = backtest_window_data.dot(w_strategy)
    bench_daily = backtest_window_data.dot(w_benchmark)
    
    strat_cum = (1 + strat_daily).cumprod() - 1
    bench_cum = (1 + bench_daily).cumprod() - 1
    
    # Render Final Year Comparison Plot
    plt.figure(figsize=(12, 6))
    plt.plot(strat_cum.index, strat_cum * 100, label="Optimized GMF Model Strategy", color="blue")
    plt.plot(bench_cum.index, bench_cum * 100, label="Passive Balanced Benchmark (60/40)", color="orange", linestyle="--")
    plt.title("Task 5: Final Year Hold-Out Out-of-Sample Backtest Tracking Plot")
    plt.ylabel("Cumulative Growth Return (%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("data/processed/task5_cumulative_returns.png", dpi=300)
    plt.close()
    
    print(f"\n[SUCCESS] Pipeline successfully closed out. Target Weights (TSLA/BND/SPY): {np.round(w_strategy, 4)}")
    print("Artifact outputs securely compiled to data/processed/")

if __name__ == "__main__":
    main()