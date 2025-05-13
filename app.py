# ====================== IMPORTS ======================
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from keras.models import load_model
import yfinance as yf
from datetime import datetime

# ====================== PAGE CONFIG (Must be FIRST Streamlit call) ======================
st.set_page_config(page_title="📊 LSTM Stock Predictor", layout="wide")

# ====================== PAGE TITLE ======================
st.title('📈 Stock Trend Prediction using LSTM')

# ====================== USER INPUT ======================
user_input = st.text_input('Enter Stock Ticker', 'AAPL')

# ====================== DATA LOADING ======================
start = datetime(2010, 1, 1)
end = datetime(2024, 1, 1)

data = yf.download(user_input, start=start, end=end)

# ====================== DATA DESCRIPTION ======================
st.subheader('📊 Raw Data')
st.write(data.tail())

# ====================== PLOTTING FUNCTIONS ======================
def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['Open'], name='Stock Open'))
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name='Stock Close'))
    fig.layout.update(title_text='Time Series Data', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()

# ====================== DATA PREPROCESSING ======================
data_training = data['Close'][0:int(len(data)*0.70)]
data_testing = data['Close'][int(len(data)*0.70):]

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

data_training_array = scaler.fit_transform(np.array(data_training).reshape(-1, 1))

# Load model
model = load_model('keras_model.h5')

# Testing data prep
past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
input_data = scaler.transform(final_df.values.reshape(-1, 1))

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)

# ====================== MODEL PREDICTION ======================
y_predicted = model.predict(x_test)

scaler = scaler.scale_
scale_factor = 1 / scaler[0]

y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

# ====================== FINAL GRAPH ======================
st.subheader('📈 Predictions vs Original')
fig2 = plt.figure(figsize=(12, 6))
plt.plot(y_test, 'b', label='Original Price')
plt.plot(y_predicted, 'r', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)



