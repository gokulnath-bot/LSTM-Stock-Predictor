# 📂 Step 1: Convert Notebook to Streamlit App (app.py)
# ================================

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import io

st.set_page_config(page_title="📊 LSTM Stock Predictor", layout="wide")

# Title
st.title("🧠📈 LSTM-Based Stock Price Predictor")

# File Upload
uploaded_file = st.file_uploader("📤 Upload your Excel file (with 'Close*' column)", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    df.rename(columns={'Close*': 'Close'}, inplace=True)
    df = df[['Close']].dropna()
    st.success("✅ File uploaded and processed successfully!")
    st.line_chart(df['Close'], use_container_width=True)

    # Normalize
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)

    # Sequence Creator
    def create_sequences(data, seq_len=60):
        X, y = [], []
        for i in range(seq_len, len(data)):
            X.append(data[i - seq_len:i])
            y.append(data[i])
        return np.array(X), np.array(y)

    seq_len = st.slider("📏 Sequence Length", min_value=30, max_value=120, value=60, step=10)
    X, y = create_sequences(scaled_data, seq_len)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # LSTM Model
    model = Sequential([
        LSTM(50, return_sequences=False, input_shape=(X_train.shape[1], 1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')

    if st.button("🚀 Train Model"):
        with st.spinner("Training in progress..."):
            model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)
        st.success("✅ Training complete!")

        # Predict
        y_pred = model.predict(X_test)
        y_pred_inv = scaler.inverse_transform(y_pred)
        y_actual_inv = scaler.inverse_transform(y_test)

        # Buy/Sell Signals
        buy_signals = []
        sell_signals = []
        for i in range(1, len(y_pred_inv)):
            if y_pred_inv[i] > y_pred_inv[i-1] and y_actual_inv[i] < y_pred_inv[i]:
                buy_signals.append(i)
            elif y_pred_inv[i] < y_pred_inv[i-1] and y_actual_inv[i] > y_pred_inv[i]:
                sell_signals.append(i)

        # Static Plot
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(y_actual_inv, label='Actual', color='blue')
        ax.plot(y_pred_inv, label='Predicted', color='orange')
        ax.scatter(buy_signals, y_pred_inv[buy_signals], label='Buy', marker='^', color='green')
        ax.scatter(sell_signals, y_pred_inv[sell_signals], label='Sell', marker='v', color='red')
        ax.set_title('Stock Price Prediction with Buy/Sell Signals')
        ax.set_xlabel('Days')
        ax.set_ylabel('Price')
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

        # Optional Animation
        if st.checkbox("🎞 Show Animated Plot"):
            def animate_predictions(y_true, y_pred):
                fig = go.Figure(
                    data=[
                        go.Scatter(x=[], y=[], name="Actual", mode='lines', line=dict(color='blue')),
                        go.Scatter(x=[], y=[], name="Predicted", mode='lines', line=dict(color='orange'))
                    ],
                    layout=go.Layout(
                        title="📊 Animated Stock Price Prediction",
                        xaxis=dict(range=[0, len(y_true)]),
                        yaxis=dict(range=[min(min(y_true), min(y_pred)) - 10, max(max(y_true), max(y_pred)) + 10]),
                        updatemenus=[dict(
                            type="buttons",
                            showactive=False,
                            buttons=[dict(label="Play", method="animate",
                                          args=[None, {"frame": {"duration": 30, "redraw": True}, "fromcurrent": True}])]
                        )]
                    ),
                    frames=[go.Frame(data=[
                        go.Scatter(x=list(range(k + 1)), y=y_true[:k + 1].flatten(), mode='lines'),
                        go.Scatter(x=list(range(k + 1)), y=y_pred[:k + 1].flatten(), mode='lines')
                    ]) for k in range(1, len(y_true))]
                )
                st.plotly_chart(fig)

            animate_predictions(y_actual_inv, y_pred_inv)

        # Evaluation
        rmse = np.sqrt(mean_squared_error(y_actual_inv, y_pred_inv))
        mae = mean_absolute_error(y_actual_inv, y_pred_inv)
# 📂 Step 2: Convert Notebook to Streamlit App (app.py)
# ================================

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import io

st.set_page_config(page_title="📊 LSTM Stock Predictor", layout="wide")

# Title
st.title("🧠📈 LSTM-Based Stock Price Predictor")

# File Upload
uploaded_file = st.file_uploader("📤 Upload your Excel file (with 'Close*' column)", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    df.rename(columns={'Close*': 'Close'}, inplace=True)
    df = df[['Close']].dropna()
    st.success("✅ File uploaded and processed successfully!")
    st.line_chart(df['Close'], use_container_width=True)

    # Normalize
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)

    # Sequence Creator
    def create_sequences(data, seq_len=60):
        X, y = [], []
        for i in range(seq_len, len(data)):
            X.append(data[i - seq_len:i])
            y.append(data[i])
        return np.array(X), np.array(y)

    seq_len = st.slider("📏 Sequence Length", min_value=30, max_value=120, value=60, step=10)
    X, y = create_sequences(scaled_data, seq_len)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # LSTM Model
    model = Sequential([
        LSTM(50, return_sequences=False, input_shape=(X_train.shape[1], 1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')

    if st.button("🚀 Train Model"):
        with st.spinner("Training in progress..."):
            model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)
        st.success("✅ Training complete!")

        # Predict
        y_pred = model.predict(X_test)
        y_pred_inv = scaler.inverse_transform(y_pred)
        y_actual_inv = scaler.inverse_transform(y_test)

        # Buy/Sell Signals
        buy_signals = []
        sell_signals = []
        for i in range(1, len(y_pred_inv)):
            if y_pred_inv[i] > y_pred_inv[i-1] and y_actual_inv[i] < y_pred_inv[i]:
                buy_signals.append(i)
            elif y_pred_inv[i] < y_pred_inv[i-1] and y_actual_inv[i] > y_pred_inv[i]:
                sell_signals.append(i)

        # Static Plot
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(y_actual_inv, label='Actual', color='blue')
        ax.plot(y_pred_inv, label='Predicted', color='orange')
        ax.scatter(buy_signals, y_pred_inv[buy_signals], label='Buy', marker='^', color='green')
        ax.scatter(sell_signals, y_pred_inv[sell_signals], label='Sell', marker='v', color='red')
        ax.set_title('Stock Price Prediction with Buy/Sell Signals')
        ax.set_xlabel('Days')
        ax.set_ylabel('Price')
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

        # Optional Animation
        if st.checkbox("🎞 Show Animated Plot"):
            def animate_predictions(y_true, y_pred):
                fig = go.Figure(
                    data=[
                        go.Scatter(x=[], y=[], name="Actual", mode='lines', line=dict(color='blue')),
                        go.Scatter(x=[], y=[], name="Predicted", mode='lines', line=dict(color='orange'))
                    ],
                    layout=go.Layout(
                        title="📊 Animated Stock Price Prediction",
                        xaxis=dict(range=[0, len(y_true)]),
                        yaxis=dict(range=[min(min(y_true), min(y_pred)) - 10, max(max(y_true), max(y_pred)) + 10]),
                        updatemenus=[dict(
                            type="buttons",
                            showactive=False,
                            buttons=[dict(label="Play", method="animate",
                                          args=[None, {"frame": {"duration": 30, "redraw": True}, "fromcurrent": True}])]
                        )]
                    ),
                    frames=[go.Frame(data=[
                        go.Scatter(x=list(range(k + 1)), y=y_true[:k + 1].flatten(), mode='lines'),
                        go.Scatter(x=list(range(k + 1)), y=y_pred[:k + 1].flatten(), mode='lines')
                    ]) for k in range(1, len(y_true))]
                )
                st.plotly_chart(fig)

            animate_predictions(y_actual_inv, y_pred_inv)

        # Evaluation
        rmse = np.sqrt(mean_squared_error(y_actual_inv, y_pred_inv))
        mae = mean_absolute_error(y_actual_inv, y_pred_inv)

        st.markdown("### 📊 Evaluation Metrics")
        st.write(f"**RMSE:** {rmse:.2f}")
        st.write(f"**MAE:** {mae:.2f}")

        st.markdown("### 📈 Strategy Tips")
        st.write("• Buy when predicted price is rising and current price is low (green ↑)")
        st.write("• Sell when predicted price is falling and current price is high (red ↓)")
else:
    st.warning("👆 Please upload an Excel file to proceed.")


