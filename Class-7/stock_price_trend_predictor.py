import streamlit as st
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import yfinance as yf

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ======================================
# Function to Fetch Stock Data
# ======================================
def fetch_stock_data(symbol):
    try:
        stock_obj = yf.Ticker(symbol)
        hist_data = stock_obj.history(period='5y')
    except Exception as e:
        st.error(f"Error fetching stock data: {e}")
        return pd.DataFrame()

    df = hist_data[['Close', 'Volume']].dropna().copy()
    df['Day'] = np.arange(len(df))
    return df

# ======================================
# Load Data
# ======================================
SYMBOL = 'AAPL'  # You can adjust this ticker symbol
data_df = fetch_stock_data(SYMBOL)

if data_df.empty:
    st.stop()

# Features and Target
X_features = data_df[['Day', 'Volume']]
y_target = data_df['Close']

# ======================================
# Split the Dataset
# ======================================
X_train, X_test, y_train, y_test = train_test_split(
    X_features, y_target, test_size=0.2, random_state=42
)

# ======================================
# Train the Linear Regression Model
# ======================================
regressor = LinearRegression()
regressor.fit(X_train, y_train)
predictions = regressor.predict(X_test)

# ======================================
# Evaluate Model Performance
# ======================================
mse_value = mean_squared_error(y_test, predictions)
r2_value = r2_score(y_test, predictions)

n_points = X_test.shape[0]
n_features = X_test.shape[1]
adjusted_r2 = 1 - (1 - r2_value) * ((n_points - 1) / (n_points - n_features - 1))

# ======================================
# Streamlit App Layout
# ======================================
st.title("📈 Stock Price Trend Analyzer")
st.write(f"### Stock Ticker: {SYMBOL}")

# Metrics Output
st.subheader("📊 Model Evaluation Metrics")
st.write(f"Mean Squared Error (MSE): {mse_value:.2f}")
st.write(f"R-squared: {r2_value:.4f}")
st.write(f"Adjusted R-squared: {adjusted_r2:.4f}")

# ======================================
# Visualization: Actual vs Predicted
# ======================================
st.subheader("📉 Price Trend Visualization")

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(data_df['Day'], data_df['Close'], label='Actual Prices', color='blue')
ax.scatter(X_test['Day'], predictions, color='red', label='Predicted Prices')
ax.set_xlabel("Day")
ax.set_ylabel("Stock Price")
ax.set_title(f"{SYMBOL} Stock Price Over Time")
ax.legend()
st.pyplot(fig)

# ======================================
# Prediction Function
# ======================================
def predict_future_price(day_value, volume_value):
    features_array = np.array([day_value, volume_value]).reshape(1, -1)
    predicted_price = regressor.predict(features_array)
    return predicted_price[0]

# ======================================
# Sidebar for User Input
# ======================================
st.sidebar.header("Predict Stock Price")

future_day = st.sidebar.number_input(
    "Enter Future Day:",
    min_value=int(data_df['Day'].min()),
    max_value=int(data_df['Day'].max()) + 30,
    value=int(data_df['Day'].max()) + 1
)

expected_volume = st.sidebar.number_input(
    "Enter Expected Volume:",
    min_value=int(data_df['Volume'].min()),
    max_value=int(data_df['Volume'].max()),
    value=int(data_df['Volume'].mean())
)

if st.sidebar.button("Predict Price"):
    future_price = predict_future_price(future_day, expected_volume)
    st.sidebar.success(f"📈 Predicted Stock Price: ${future_price:.2f}")
