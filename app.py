import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import plotly.graph_objs as go

# 🌐 Streamlit Page Config — must be FIRST!
st.set_page_config(page_title="📊 LSTM Stock Predictor", layout="wide")

# 🎨 App Title
st.title("📈 LSTM Stock Price Predictor")
st.markdown("Upload a Yahoo Finance `.csv` file to predict future stock prices using an LSTM model.")

# 📤 File Uploader
uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])

# ⚙️ Helper to clean column names
def clean_column_names(df):
    df.columns = df.columns.str.replace(r'[^\w\s]', '', regex=True)
    if 'Close' not in df.columns:
        if 'Adj Close' in df.columns:
            df.rename(columns={'Adj Close': 'Close'}, inplace=True)
    return df

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df = clean_column_names(df)

    if 'Close' not in df.columns:
        st.error("❌ The file must contain a 'Close' column.")
        st.stop()

    st.success("✅ File loaded successfully!")

    # 📊 Show raw data
    st.subheader("Preview of Uploaded Data")
    st.write(df.tail())

    # 📈 Prepare Data
    data = df.filter(['Close'])
    dataset = data.values

    training_data_len = int(np.ceil(len(dataset) * .8))

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    train_data = scaled_data[0:training_data_len, :]
    x_train, y_train = [], []

    for i in range(60, len(train_data)):
        x_train.append(train_data[i-60:i, 0])
        y_train.append(train_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # 🤖 Build the LSTM Model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))

    # 🧠 Train
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, batch_size=1, epochs=1)

    # 📈 Predicting
    test_data = scaled_data[training_data_len - 60:, :]
    x_test, y_test = [], dataset[training_data_len:, :]

    for i in range(60, len(test_data)):
        x_test.append(test_data[i-60:i, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    # 🧮 RMSE
    rmse = np.sqrt(np.mean((predictions - y_test) ** 2))

    st.subheader(f"📉 Root Mean Squared Error (RMSE): {rmse:.2f}")

    # 📊 Plot Results
    train = data[:training_data_len]
    valid = data[training_data_len:]
    valid['Predictions'] = predictions

    st.subheader("📊 Predicted vs Actual")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train.index, y=train['Close'], name='Train'))
    fig.add_trace(go.Scatter(x=valid.index, y=valid['Close'], name='Actual'))
    fig.add_trace(go.Scatter(x=valid.index, y=valid['Predictions'], name='Predicted'))
    st.plotly_chart(fig, use_container_width=True)

    st.write("✅ Done! Try uploading a different file or retraining the model.")
else:
    st.info("📥 Please upload a CSV file to continue.")
