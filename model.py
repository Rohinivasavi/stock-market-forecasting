from prophet import Prophet
import pandas as pd

def train_prophet_model(df):
    # Ensure required columns exist
    if "Date" not in df.columns:
        raise ValueError("Date column not found in dataframe.")

    # Prefer 'Close', fallback on 'Adj Close'
    price_col = "Close"
    if "Close" not in df.columns:
        if "Adj Close" in df.columns:
            price_col = "Adj Close"
        else:
            raise ValueError("Neither 'Close' nor 'Adj Close' found in dataframe.")

    # Select only required columns
    df = df[["Date", price_col]].copy()

    # Rename for Prophet
    df = df.rename(columns={"Date": "ds", price_col: "y"})

    df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")

    df = df.dropna(subset=["ds", "y"])
    df = df.sort_values("ds")

    model = Prophet(daily_seasonality=True)
    model.fit(df)

    return model


def forecast_future(model, days):
    future = model.make_future_dataframe(periods=days)
    forecast = model.predict(future)
    return forecast
