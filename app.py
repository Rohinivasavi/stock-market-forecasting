import streamlit as st
import plotly.graph_objects as go
from fetch_data import get_stock_data
from model import train_prophet_model, forecast_future

st.title("📈 Stock Market Forecasting Dashboard")

symbol = st.text_input("Enter Stock Symbol (Example: AAPL, TSLA, INFY.NS)", "AAPL")

if st.button("Fetch Data"):
    data = get_stock_data(symbol)

    # DEBUG LINE TO CHECK COLUMNS
    st.write("Columns:", data.columns)

    st.subheader("📌 Historical Data")
    st.write(data.tail())

    # Plot historical
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data["Date"], y=data["Close"], mode="lines", name="Close Price"))
    st.plotly_chart(fig)

    # Train model
    st.subheader("⏳ Training Prediction Model...")
    model = train_prophet_model(data)

    days = st.slider("Select days to forecast:", 30, 365)
    forecast = forecast_future(model, days)

    st.subheader("🔮 Forecasted Chart")
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat"], mode="lines", name="Prediction"))
    st.plotly_chart(fig2)

    st.subheader("📄 Forecast Data")
    st.write(forecast.tail())
