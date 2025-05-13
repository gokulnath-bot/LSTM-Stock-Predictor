import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
import io

# 🛠 PAGE CONFIG
st.set_page_config(page_title="📊 LSTM Stock Predictor", layout="wide")

# 🖼️ Title
st.title("📈 LSTM Stock Price Predictor")
st.markdown("""
This app predicts stock prices using an LSTM (Long Short-Term Memory) model.
Upload your CSV file containing a 'Close' column to get started.
""")

# 📤 File Upload
uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("🔍 Data Preview")
    st.dataframe(df.head())
    st.write("Available Columns:", df.columns.tolist())

    if 'Close' not in df.columns:
        st.error("❌ The file must contain a 'Close' column.")
        st.stop()

    # 📊 Data Preparation
    dataset = df[['Close']].values
    training_data_len = int(np.ceil(len(dataset) * 0.8))

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    train_data = scaled_data[0:int(training_data_len), :]
    x_train = []
    y_train = []

    for i in range(60, len(train_data)):
        x_train.append(train_data[i - 60:i, 0])
        y_train.append(train_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # 🧠 LSTM Model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, batch_size=1, epochs=1)

    # 🧪 Testing the Model
    test_data = scaled_data[training_data_len - 60:, :]
    x_test = []
    y_test = dataset[training_data_len:, :]

    for i in range(60, len(test_data)):
        x_test.append(test_data[i - 60:i, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    # 📉 Plot the Results
    train = df[:training_data_len]
    valid = df[training_data_len:]
    valid['Predictions'] = predictions

    st.subheader("📉 Predicted vs Actual")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train.index, y=train['Close'], name="Training Data"))
    fig.add_trace(go.Scatter(x=valid.index, y=valid['Close'], name="Actual"))
    fig.add_trace(go.Scatter(x=valid.index, y=valid['Predictions'], name="Predicted"))
    st.plotly_chart(fig, use_container_width=True)

    # 📥 Download Prediction
    csv = valid[['Close', 'Predictions']].to_csv(index=False)
    st.download_button(
        label="📥 Download Predictions as CSV",
        data=csv,
        file_name='predictions.csv',
        mime='text/csv',
    )
else:
    st.info("👈 Upload a CSV file to begin prediction.")




