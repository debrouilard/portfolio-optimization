# 📈 Portfolio Optimization using Time Series Forecasting

## Project Overview

This project was developed as part of the 10 Academy Week 9 Challenge.

The objective is to analyze historical financial market data, build forecasting models, and optimize an investment portfolio using Modern Portfolio Theory (MPT).

The project focuses on three assets:

- TSLA (Tesla)
- SPY (S&P 500 ETF)
- BND (Vanguard Total Bond Market ETF)

Historical data is obtained from Yahoo Finance using the yfinance Python library.

---

## Objectives

- Download historical market data
- Clean and preprocess the data
- Perform Exploratory Data Analysis (EDA)
- Detect outliers
- Test stationarity
- Calculate investment risk metrics
- Build ARIMA and LSTM forecasting models
- Forecast future prices
- Optimize a portfolio
- Backtest the investment strategy

---

## Project Structure

```
portfolio-optimization/

│
├── .github/
│ └── workflows/
│ └── unittests.yml
│
├── data/
│ ├── raw/
│ └── processed/
│
├── notebooks/
│ └── 01_EDA.ipynb
│
├── scripts/
│ ├── download_data.py
│ ├── preprocess.py
│ ├── eda.py
│ └── train_arima.py
│
├── src/
│ ├── data_loader.py
│ ├── preprocessing.py
│ ├── statistics.py
│ ├── visualization.py
│ ├── forecasting.py
│ └── utils.py
│
├── tests/
│ └── test_preprocessing.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Dataset

Source:

Yahoo Finance (yfinance)

Assets:

- TSLA
- SPY
- BND

Time Period:

January 1, 2015 – June 30, 2026

---

## Installation

Clone the repository

```bash
git clone https://github.com/yourusername/portfolio-optimization.git
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Project

Download data

```bash
python scripts/download_data.py
```

Preprocess data

```bash
python scripts/preprocess.py
```

Run exploratory analysis

```bash
python scripts/eda.py
```

Train ARIMA model

```bash
python scripts/train_arima.py
```

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- yfinance
- Statsmodels
- pmdarima
- TensorFlow
- Scikit-learn

---

## Risk Metrics

The project computes:

- Daily Returns
- Rolling Mean
- Rolling Volatility
- Value at Risk (VaR)
- Sharpe Ratio

---

## Forecasting Models

- ARIMA
- LSTM

---

## Portfolio Optimization

The final stage uses Modern Portfolio Theory (MPT) to determine optimal asset allocation.

---

## Author

Blen Debebe

10 Academy Week 9 Challenge