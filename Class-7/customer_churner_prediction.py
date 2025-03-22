import streamlit as st
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ================================
# Data Loading & Preprocessing
# ================================

@st.cache_data
def load_customer_data():
    try:
        df_raw = pd.read_csv('customer_churn_data.csv')
    except FileNotFoundError:
        st.error("Data file 'customer_churn_data.csv' not found.")
        st.stop()

    df = df_raw.copy()

    # Drop unnecessary columns
    if 'customerID' in df.columns:
        df.drop(columns=['customerID'], inplace=True)

    # Convert TotalCharges to numeric
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(inplace=True)

    # Binary encoding for Yes/No columns
    binary_columns = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
    for col in binary_columns:
        df[col] = df[col].map({'Yes': 1, 'No': 0})

    # Encode target
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    # One-hot encoding for categorical features
    categorical_columns = ['gender', 'MultipleLines', 'InternetService',
                           'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                           'TechSupport', 'StreamingTV', 'StreamingMovies',
                           'Contract', 'PaymentMethod']

    df_encoded = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

    X_features = df_encoded.drop(columns=['Churn'])
    y_target = df_encoded['Churn']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_features)

    return df_encoded, X_features, X_scaled, y_target, scaler

# ================================
# Train Models Function
# ================================

def train_churn_models(X_scaled, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # Logistic Regression
    log_reg = LogisticRegression(max_iter=1000, random_state=42)
    log_reg.fit(X_train, y_train)
    y_pred_log = log_reg.predict(X_test)

    mse_log = np.mean((y_test - y_pred_log) ** 2)
    r2_log = log_reg.score(X_test, y_test)

    n_samples, n_features = X_test.shape
    adj_r2_log = 1 - (1 - r2_log) * (n_samples - 1) / (n_samples - n_features - 1)

    # SGD Classifier
    sgd_clf = SGDClassifier(loss='log_loss', max_iter=1000, tol=1e-3, random_state=42)
    sgd_clf.fit(X_train, y_train)
    y_pred_sgd = sgd_clf.predict(X_test)
    sgd_acc = accuracy_score(y_test, y_pred_sgd)

    return {
        'logistic_model': log_reg,
        'sgd_model': sgd_clf,
        'y_test': y_test,
        'y_pred_log': y_pred_log,
        'mse_log': mse_log,
        'r2_log': r2_log,
        'adj_r2_log': adj_r2_log,
        'sgd_accuracy': sgd_acc,
        'X_test': X_test
    }

# ================================
# Load Data & Train Models
# ================================
df_encoded, X_df, X_scaled, y_target, scaler_obj = load_customer_data()
results = train_churn_models(X_scaled, y_target)

# ================================
# Streamlit App Layout
# ================================

st.title("📊Customer Churn Prediction Tool")

# Model Evaluation
st.header("📈 Model Evaluation Metrics")

st.subheader("Logistic Regression")
st.write(f"Mean Squared Error (MSE): {results['mse_log']:.4f}")
st.write(f"R-squared: {results['r2_log']:.4f}")
st.write(f"Adjusted R-squared: {results['adj_r2_log']:.4f}")

st.subheader("SGD Classifier")
st.write(f"Accuracy: {results['sgd_accuracy']:.4f}")

# Classification Report
st.subheader("Classification Report (Logistic Regression)")
report_df = pd.DataFrame(classification_report(
    results['y_test'], results['y_pred_log'], output_dict=True
)).transpose()
st.dataframe(report_df)

# Coefficient Analysis
st.subheader("Feature Importance (Logistic Regression Coefficients)")
coef_df = pd.DataFrame({
    'Feature': X_df.columns,
    'Coefficient': results['logistic_model'].coef_[0]
}).sort_values(by='Coefficient', ascending=False)
st.dataframe(coef_df)

# Confusion Matrix
st.subheader("Confusion Matrix (Logistic Regression)")
conf_matrix = confusion_matrix(results['y_test'], results['y_pred_log'])

fig, ax = plt.subplots(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Churn', 'Churn'],
            yticklabels=['No Churn', 'Churn'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
st.pyplot(fig)

# ================================
# Sidebar: Churn Prediction
# ================================

st.sidebar.header("Predict Customer Churn")

input_values = {}
for column in X_df.columns:
    input_values[column] = st.sidebar.number_input(
        f"{column}",
        float(X_df[column].min()),
        float(X_df[column].max()),
        float(X_df[column].mean())
    )

if st.sidebar.button("Predict Churn"):
    user_input_df = pd.DataFrame([input_values])
    scaled_user_input = scaler_obj.transform(user_input_df)

    churn_prediction = results['logistic_model'].predict(scaled_user_input)[0]

    if churn_prediction == 1:
        st.sidebar.error("⚠️ This customer is likely to churn!")
    else:
        st.sidebar.success("✅ This customer is unlikely to churn.")
