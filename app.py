import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import plotly.graph_objects as go

# 🛠 Set Streamlit page config
st.set_page_config(page_title="📊 LSTM Stock Predictor", layout="wide")

# ✨ Title
st.title("📈 LSTM Stock Price Predictor")
st.markdown("Upload a CSV file containing historical stock data with at least a 'Close' or 'Adj Close' column.")

# 📁 Upload section
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

# 🧹 Clean column names
def clean_column_names(df):
    df.columns = [col.strip().replace("*", "").replace("**", "") for col in df.columns]
    return df

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df = clean_column_names(df)

    if 'Close' not in df.columns:
        if 'Adj Close' in df.columns:
            df.rename(columns={'Adj Close': 'Close'}, inplace=True)

    try:
        # Remove commas and convert to float for proper numerical handling
        df['Close'] = df['Close'].astype(str).str.replace(',', '').astype(float)
    except Exception as e:
        st.error(f"❌ Error converting 'Close' column to float: {e}")
        st.stop()

    st.subheader("Raw Data")
    st.dataframe(df.tail(10))

    # ✂️ Prepare data
    data = df.filter(['Close'])
    dataset = data.values
    training_data_len = int(np.ceil(len(dataset) * .8))

    # 📏 Scale data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    # 🧱 Training data
    train_data = scaled_data[0:int(training_data_len), :]
    x_train = []
    y_train = []

    for i in range(60, len(train_data)):
        x_train.append(train_data[i-60:i, 0])
        y_train.append(train_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # 🤖 Build model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))

    # 🏋️‍♂️ Train
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, batch_size=1, epochs=1)

    # 📊 Test data
    test_data = scaled_data[training_data_len - 60:, :]
    x_test = []
    y_test = dataset[training_data_len:, :]

    for i in range(60, len(test_data)):
        x_test.append(test_data[i-60:i, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # 🔮 Predictions
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    # 📉 Plot
    train = data[:training_data_len]
    valid = data[training_data_len:]
    valid = valid.assign(Predictions=predictions.flatten())

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train.index, y=train['Close'], name='Train Close'))
    fig.add_trace(go.Scatter(x=valid.index, y=valid['Close'], name='Actual Close'))
    fig.add_trace(go.Scatter(x=valid.index, y=valid['Predictions'], name='Predicted Close'))
    fig.update_layout(title='Stock Price Prediction vs Actual', xaxis_title='Date', yaxis_title='Price')

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Prediction Data")
    st.dataframe(valid.tail(10))
else:
    st.info("Please upload a CSV file with a 'Close' or 'Adj Close' column to begin.")

