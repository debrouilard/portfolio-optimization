import os
import sys
import pandas as pd

# Structural alignment to root path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.forecasting import (
    split_data_chronologically, 
    optimize_arima_params, 
    generate_arima_forecast, 
    evaluate_forecast_metrics
)

def main():
    processed_data_path = "data/processed/cleaned_data.csv"
    
    if not os.path.exists(processed_data_path):
        print(f"[ERROR] Processed data file missing at {processed_data_path}. Please execute preprocess.py first.")
        return

    # Ingest cleaned historical data frame
    df = pd.read_csv(processed_data_path, index_col=0, parse_dates=True)
    
    # Target specific high-growth tracking asset column (TSLA Close/Adj Close)
    # Assumes column naming structure format generated from task 1 is 'Close_TSLA'
    target_asset = "Close_TSLA"
    
    if target_asset not in df.columns:
        print(f"[ERROR] Target column '{target_asset}' not found in the preprocessed dataset.")
        return

    print(f"\n=== Initializing Task 2 Modeling Framework for {target_asset} ===")
    
    # Step 1: Chronological Train/Test Split (Preserving Temporal Order)
    train, test = split_data_chronologically(df, target_col=target_asset, split_date="2025-01-01")
    print(f"-> Training Dataset Range: {train.index.min().date()} to {train.index.max().date()} ({len(train)} steps)")
    print(f"-> Testing Dataset Range:  {test.index.min().date()} to {test.index.max().date()} ({len(test)} steps)")

    # Step 2: Auto-Arima Parameter Optimization Search
    optimal_order = optimize_arima_params(train, seasonal=False)
    
    # Step 3: Train and Forecast Generation
    forecast = generate_arima_forecast(train, test, order=optimal_order)
    
    # Step 4: Quantify Performance Metrics
    metrics = evaluate_forecast_metrics(test, forecast)
    
    # Output metrics directly to terminal console
    print("\n==================================================")
    print("      ARIMA FORECAST MODEL PERFORMANCE SUMMARY    ")
    print("==================================================")
    print(f"Optimal Parameters Fitted : ARIMA{optimal_order}")
    print(f"Mean Absolute Error (MAE) : {metrics['MAE']:.4f}")
    print(f"Root Mean Sq. Error (RMSE): {metrics['RMSE']:.4f}")
    print(f"Mean Abs. % Error (MAPE)  : {metrics['MAPE']:.2f}%")
    print("==================================================")

if __name__ == "__main__":
    main()