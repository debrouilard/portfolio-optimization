import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- TASK 2: Chronological Splitting ---
def split_data_chronologically(df: pd.DataFrame, target_col: str, split_date: str = "2025-01-01"):
    """Splits dataset chronologically to preserve temporal order without shuffling."""
    train = df[df.index < split_date][target_col]
    test = df[df.index >= split_date][target_col]
    return train, test

# --- TASK 2: ARIMA Track ---
def optimize_and_forecast_arima(train: pd.Series, test: pd.Series):
    """Optimizes ARIMA parameters via auto_arima and generates out-of-sample test forecasts."""
    logging.info("Optimizing ARIMA parameters...")
    order = auto_arima(train, seasonal=False, stepwise=True, suppress_warnings=True, error_action="ignore").order
    logging.info(f"Optimal ARIMA Order Selected: {order}")
    
    model = ARIMA(train, order=order)
    fitted_model = model.fit()
    forecast = fitted_model.forecast(steps=len(test))
    return pd.Series(forecast, index=test.index), fitted_model

# --- TASK 2: LSTM Track (Missing Implementation Added) ---
def prepare_lstm_sequences(series: pd.Series, window_size: int = 60):
    """Transforms a time series into windowed sequences (X) and targets (y)."""
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(series.values.reshape(-1, 1))
    
    X, y = [], []
    for i in range(len(scaled_data) - window_size):
        X.append(scaled_data[i:(i + window_size)])
        y.append(scaled_data[i + window_size])
    return np.array(X), np.array(y), scaler

def train_and_forecast_lstm(train: pd.Series, test: pd.Series, window_size: int = 60, epochs=15, batch_size=32):
    """Builds, trains, and runs a sequential LSTM framework for out-of-sample testing."""
    logging.info("Preparing windowed sequences and training LSTM architecture...")
    X_train, y_train, scaler = prepare_lstm_sequences(train, window_size)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=(window_size, 1)),
        Dropout(0.2),
        LSTM(units=50, return_sequences=False),
        Dropout(0.2),
        Dense(units=1)
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    
    # Generate Multi-Step out-of-sample test forecast recursively
    full_inputs = pd.concat((train[-window_size:], test))
    scaled_inputs = scaler.transform(full_inputs.values.reshape(-1, 1))
    
    predictions = []
    current_window = list(scaled_inputs[:window_size].flatten())
    
    for i in range(len(test)):
        X_test = np.array(current_window[-window_size:]).reshape(1, window_size, 1)
        pred_scaled = model.predict(X_test, verbose=0)[0, 0]
        predictions.append(pred_scaled)
        current_window.append(scaled_inputs[window_size + i][0]) # Teacher forcing for validation comparison
        
    pred_prices = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    return pd.Series(pred_prices.flatten(), index=test.index), model, scaler

# --- TASK 2: Multi-Model Evaluation Matrix ---
def calculate_metrics(actual, predicted):
    """Computes comparative metrics tracking MAE, RMSE, and MAPE."""
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}