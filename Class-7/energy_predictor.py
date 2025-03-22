import streamlit as st
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# ==========================
# Load and Prepare Dataset
# ==========================
DATA_PATH = "owid-energy-data.csv"
try:
    dataset = pd.read_csv(DATA_PATH)
except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()

# ==========================
# Streamlit App Configuration
# ==========================
st.title("🔋 Energy Usage Exploration & Forecast")
st.write("Available columns:", dataset.columns.tolist())

# ==========================
# Determine Target Feature
# ==========================
consumption_cols = [col for col in dataset.columns if 'consumption' in col.lower()]
if not consumption_cols:
    st.error("No energy consumption column found.")
    st.stop()

target_col = consumption_cols[0]
st.success(f"Targeting column: '{target_col}'")

# ==========================
# Data Cleaning
# ==========================
data = dataset.copy()
data.dropna(inplace=True)

categorical_vars = data.select_dtypes(include=['object']).columns.tolist()
if categorical_vars:
    data = pd.get_dummies(data, columns=categorical_vars, drop_first=True)

# ==========================
# Feature Inspection
# ==========================
time_vars = [c for c in data.columns if 'year' in c.lower() or 'month' in c.lower()]
temp_vars = [c for c in data.columns if 'temperature' in c.lower()]

st.write("Time-based features:", time_vars)
st.write("Temperature-related features:", temp_vars)

# ==========================
# Exploratory Data Analysis
# ==========================
st.header("📊 Data Overview & EDA")

st.subheader("Summary Statistics")
st.dataframe(data.describe())

st.subheader("Correlation Heatmap")
fig1, ax1 = plt.subplots(figsize=(10, 6))
sns.heatmap(data.corr(), cmap='coolwarm', ax=ax1)
st.pyplot(fig1)

if time_vars:
    st.subheader("Energy Consumption Over Time")
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sorted_data = data.sort_values(by=time_vars[0])
    ax2.plot(sorted_data[time_vars[0]], sorted_data[target_col], marker='o')
    ax2.set_xlabel(time_vars[0])
    ax2.set_ylabel(target_col)
    st.pyplot(fig2)

# ==========================
# Prepare Features & Target
# ==========================
features = data.drop(columns=[target_col])
labels = data[target_col]

scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# ==========================
# Train/Test Split
# ==========================
X_train, X_test, y_train, y_test = train_test_split(
    features_scaled, labels, test_size=0.2, random_state=42
)

# ==========================
# Train Models
# ==========================
ols = LinearRegression()
ols.fit(X_train, y_train)
y_pred_ols = ols.predict(X_test)

sgd = SGDRegressor(max_iter=1000, tol=1e-3, random_state=42)
sgd.fit(X_train, y_train)
y_pred_sgd = sgd.predict(X_test)

# ==========================
# Evaluation
# ==========================
metrics = {
    'OLS': {
        'MSE': mean_squared_error(y_test, y_pred_ols),
        'R2': r2_score(y_test, y_pred_ols)
    },
    'SGD': {
        'MSE': mean_squared_error(y_test, y_pred_sgd),
        'R2': r2_score(y_test, y_pred_sgd)
    }
}

st.header("📈 Model Performance")

for model, scores in metrics.items():
    st.subheader(f"{model} Results")
    st.write(f"Mean Squared Error: {scores['MSE']:.2f}")
    st.write(f"R-squared: {scores['R2']:.2f}")

# ==========================
# Actual vs Predicted Plot
# ==========================
st.subheader("📉 Actual vs Predicted (OLS)")
fig3, ax3 = plt.subplots()
ax3.scatter(y_test, y_pred_ols, alpha=0.6)
ax3.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
ax3.set_xlabel("Actual Consumption")
ax3.set_ylabel("Predicted Consumption")
st.pyplot(fig3)

# ==========================
# Sidebar User Prediction
# ==========================
st.sidebar.header("Make a Prediction")
user_vals = {}

for col in features.columns:
    min_val = float(data[col].min())
    max_val = float(data[col].max())
    mean_val = float(data[col].mean())
    user_vals[col] = st.sidebar.number_input(
        label=col,
        min_value=min_val,
        max_value=max_val,
        value=mean_val
    )

if st.sidebar.button("Predict Energy Consumption"):
    user_input_array = np.array([user_vals[col] for col in features.columns]).reshape(1, -1)
    scaled_input = scaler.transform(user_input_array)

    ols_pred = ols.predict(scaled_input)[0]
    sgd_pred = sgd.predict(scaled_input)[0]

    st.sidebar.success(f"OLS Prediction: {ols_pred:.2f} kWh")
    st.sidebar.info(f"SGD Prediction: {sgd_pred:.2f} kWh")
