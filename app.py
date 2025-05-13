import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np

# Streamlit page configuration
st.set_page_config(page_title="📊 Stock Price Prediction", layout="wide")

# File upload widget
uploaded_file = st.file_uploader("Upload your Yahoo Finance CSV", type=["csv"])

if uploaded_file is not None:
    # Read the CSV file
    df = pd.read_csv(uploaded_file)

    # Display the first few rows to confirm the file looks correct
    st.write(df.head())  # This will show a preview of the uploaded file

    # Check if 'Close' column exists
    if 'Close' not in df.columns:
        st.error("❌ The file must contain a 'Close' column.")
        st.stop()  # Stop execution if the 'Close' column is missing

    # Continue with data processing and predictions
    st.success("✅ 'Close' column found, proceeding with the analysis...")

    # Data Preprocessing
    df = df[['Date', 'Close']]  # Use 'Date' and 'Close' columns
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    # Scale the 'Close' values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[['Close']])

    # Show a plot of the 'Close' prices
    st.subheader("📈 Close Price Plot")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close'))
    st.plotly_chart(fig)

    # Prepare data for LSTM model
    def create_dataset(data, time_step=1):
        X, y = [], []
        for i in range(len(data) - time_step - 1):
            X.append(data[i:(i + time_step), 0])
            y.append(data[i + time_step, 0])
        return np.array(X), np.array(y)

    # Create datasets for training and testing
    time_step = 60
    X, y = create_dataset(scaled_data, time_step)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    # Now you can proceed with training your LSTM model, prediction, etc.
    # Here you can add your LSTM model code (for example, creating and fitting the model)

    # Model prediction and plotting results can go here

else:
    st.info("Please upload a Yahoo Finance CSV file to get started.")
