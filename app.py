import streamlit as st
st.set_page_config(page_title="📊 LSTM Stock Predictor", layout="wide")  # MUST be first Streamlit command

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import yfinance as yf
import datetime

# Title
st.title("📈 LSTM Stock Price Predictor")

# Sidebar
st.sidebar.header("Configuration")
selected_stock = st.sidebar.text_input("Enter stock symbol (e.g., AAPL)", "AAPL")
start_date = st.sidebar.date_input("Start date", datetime.date(2010, 1, 1))
end_date = st.sidebar.date_input("End date", datetime.date.today())

if st.sidebar.button("🔍 Fetch Data"):
    with st.spinner("Loading stock data..."):
        df = yf.download(selected_stock, start=start_date, end=end_date)

        if df.empty:
            st.error("❌ No data found for this symbol.")
        else:
            st.success("✅ Data loaded!")
            st.subheader(f"Stock data for {selected_stock}")
            st.dataframe(df.tail())

            # Plot closing price
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name="Closing Price"))
            fig.update_layout(title="Stock Closing Price Over Time", xaxis_title="Date", yaxis_title="Price")
            st.plotly_chart(fig, use_container_width=True)

            # Data preprocessing
            data = df.filter(['Close'])
            dataset = data.values
            training_data_len = int(np.ceil(len(dataset) * 0.8))

            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(dataset)

            train_data = scaled_data[0:int(training_data_len), :]
            x_train, y_train = [], []
            for i in range(60, len(train_data)):
                x_train.append(train_data[i-60:i, 0])
                y_train.append(train_data[i, 0])

            x_train, y_train = np.array(x_train), np.array(y_train)
            x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

            # Load model
            model = load_model("keras_model.h5")

            # Prepare test data
            test_data = scaled_data[training_data_len - 60:, :]
            x_test = []
            for i in range(60, len(test_data)):
                x_test.append(test_data[i-60:i, 0])

            x_test = np.array(x_test)
            x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

            # Predictions
            predictions = model.predict(x_test)
            predictions = scaler.inverse_transform(predictions)

            # Plot predictions
            valid = data[training_data_len:]
            valid['Predictions'] = predictions

            st.subheader("📊 Predicted vs Actual Closing Prices")
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=valid.index, y=valid['Close'], name="Actual Price"))
            fig2.add_trace(go.Scatter(x=valid.index, y=valid['Predictions'], name="Predicted Price"))
            fig2.update_layout(title="Model Prediction vs Actual", xaxis_title="Date", yaxis_title="Price")
            st.plotly_chart(fig2, use_container_width=True)




