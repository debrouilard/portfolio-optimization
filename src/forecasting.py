import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error, mean_squared_error
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def split_data_chronologically(df: pd.DataFrame, target_col: str, split_date: str = "2025-01-01"):
    """
    Splits the dataset chronologically by date to preserve temporal order.
    Ensures no random shuffling occurs.
    """
    logging.info(f"Splitting data chronologically at {split_date}...")
    train = df[df.index < split_date][target_col]
    test = df[df.index >= split_date][target_col]
    return train, test

def optimize_arima_params(series: pd.Series, seasonal: bool = False, m: int = 1):
    """
    Uses auto_arima to discover optimal (p, d, q) or (P, D, Q, m) parameters
    based on the Akaike Information Criterion (AIC).
    """
    logging.info("Running auto_arima parameter optimization grid search...")
    model_search = auto_arima(
        series, 
        seasonal=seasonal, 
        m=m,
        stepwise=True, 
        suppress_warnings=True, 
        error_action="ignore"
    )
    logging.info(f"Optimal Order Identified: {model_search.order}")
    return model_search.order

def generate_arima_forecast(train_series: pd.Series, test_series: pd.Series, order: tuple):
    """
    Fits the ARIMA model on the training set and projects forecasts 
    across the length of the test period.
    """
    logging.info(f"Fitting ARIMA model with order {order}...")
    model = ARIMA(train_series, order=order)
    fitted_model = model.fit()
    
    # Forecast for the length of the test set
    forecast_values = fitted_model.forecast(steps=len(test_series))
    forecast_series = pd.Series(forecast_values, index=test_series.index)
    return forecast_series

def evaluate_forecast_metrics(actual: pd.Series, predicted: pd.Series) -> dict:
    """
    Calculates operational time series performance metrics: MAE, RMSE, and MAPE.
    """
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    
    return {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape
    }