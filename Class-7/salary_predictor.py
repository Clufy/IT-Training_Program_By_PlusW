import streamlit as st
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# ===========================
# Data Loading & Preprocessing
# ===========================

@st.cache_data
def fetch_data():
    try:
        raw_data = pd.read_csv('salary_data.csv')
    except FileNotFoundError:
        st.error("File 'salary_data.csv' not found.")
        st.stop()

    data = raw_data.dropna().copy()

    cat_cols = ['experience_level', 'employment_type', 'job_title',
                'company_location', 'company_size']

    data_encoded = pd.get_dummies(data, columns=cat_cols, drop_first=True)

    X_data = data_encoded.drop(columns=['salary_in_usd', 'salary', 'salary_currency'])
    y_data = data_encoded['salary_in_usd']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_data)

    return data, X_data, X_scaled, y_data, scaler

raw_data, X_df, X_scaled_data, y_target, scaler_obj = fetch_data()

# ===========================
# Model Training Function
# ===========================
def model_training(X_data_scaled, y_data):
    X_train, X_test, y_train, y_test = train_test_split(
        X_data_scaled, y_data, test_size=0.2, random_state=42
    )

    linear_reg = LinearRegression()
    linear_reg.fit(X_train, y_train)
    pred_lr = linear_reg.predict(X_test)

    mse_lr = mean_squared_error(y_test, pred_lr)
    r2_lr = r2_score(y_test, pred_lr)

    n_samples, n_features = X_test.shape
    adj_r2_lr = 1 - (1 - r2_lr) * (n_samples - 1) / (n_samples - n_features - 1)

    ridge_reg = Ridge(alpha=1.0)
    ridge_cv_scores = cross_val_score(ridge_reg, X_train, y_train, cv=5)

    lasso_reg = Lasso(alpha=10, max_iter=5000)
    lasso_cv_scores = cross_val_score(lasso_reg, X_train, y_train, cv=5)

    return {
        'linear_model': linear_reg,
        'y_actual': y_test,
        'y_predicted': pred_lr,
        'mse': mse_lr,
        'r2': r2_lr,
        'adj_r2': adj_r2_lr,
        'ridge_cv_mean': ridge_cv_scores.mean(),
        'lasso_cv_mean': lasso_cv_scores.mean()
    }

trained_models = model_training(X_scaled_data, y_target)

# ===========================
# Streamlit Layout
# ===========================

st.title("📊 Data Science Salary Insights & Prediction")

# Evaluation Metrics
st.header("📈 Model Performance Metrics")

st.subheader("Linear Regression Results")
st.write(f"MSE: {trained_models['mse']:.2f}")
st.write(f"R-squared: {trained_models['r2']:.4f}")
st.write(f"Adjusted R-squared: {trained_models['adj_r2']:.4f}")

st.subheader("Ridge Regression (Cross-Validated)")
st.write(f"Average CV Score: {trained_models['ridge_cv_mean']:.4f}")

st.subheader("Lasso Regression (Cross-Validated)")
st.write(f"Average CV Score: {trained_models['lasso_cv_mean']:.4f}")

# Scatter Plot: Actual vs Predicted
st.header("📉 Actual vs Predicted Salaries (Linear Regression)")

fig, axis = plt.subplots(figsize=(8, 6))
sns.scatterplot(x=trained_models['y_actual'], y=trained_models['y_predicted'], ax=axis)
axis.set_xlabel("Actual Salary (USD)")
axis.set_ylabel("Predicted Salary (USD)")
axis.set_title("Actual vs Predicted Salaries")
st.pyplot(fig)

# ===========================
# User Input & Prediction
# ===========================

st.sidebar.header("Predict Your Salary")

user_inputs = {}
for column in X_df.columns:
    if 'work_year' in column:
        user_inputs[column] = st.sidebar.slider(f"{column}", min_value=2021, max_value=2023, value=2023)
    else:
        user_inputs[column] = st.sidebar.number_input(f"{column}", value=0)

if st.sidebar.button("Estimate Salary"):
    input_df = pd.DataFrame([user_inputs])
    scaled_input = scaler_obj.transform(input_df)

    salary_prediction = trained_models['linear_model'].predict(scaled_input)
    st.sidebar.success(f"💰 Estimated Salary: ${salary_prediction[0]:,.2f}")

