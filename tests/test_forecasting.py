import pytest
import pandas as pd
import numpy as np
from src.forecasting import split_data_chronologically

def test_split_data_chronologically():
    # Setup dummy date-indexed dataframe
    dates = pd.date_range(start="2024-12-25", periods=10, freq="D")
    data = {"Close_TSLA": np.arange(10)}
    df = pd.DataFrame(data, index=dates)
    
    # Split exactly at 2025-01-01
    train, test = split_data_chronologically(df, target_col="Close_TSLA", split_date="2025-01-01")
    
    # Assertions to ensure split order and data mapping integrity
    assert all(train.index < "2025-01-01")
    assert all(test.index >= "2025-01-01")
    assert len(train) == 7
    assert len(test) == 3