import yfinance as yf
import pandas as pd

def get_stock_data(symbol, period="5y", interval="1d"):
    data = yf.download(symbol, period=period, interval=interval)

    # Reset index to make Date a column
    data = data.reset_index()

    # FLATTEN MULTI-INDEX COLUMNS
    data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]

    # Clean missing values
    if "Close" in data.columns:
        data = data.dropna(subset=["Close"])
    elif "Adj Close" in data.columns:
        data = data.dropna(subset=["Adj Close"])

    return data
