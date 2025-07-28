import streamlit as st
import pandas as pd
import numpy as np
import shap
from sklearn.ensemble import RandomForestClassifier

# --------------------------------
# Streamlit Config
# --------------------------------
st.set_page_config(page_title="Churn Predictor", page_icon="📉", layout="wide")

st.markdown("""
    <style>
        body {
            background-color: #0b1d3a;
            color: white;
        }
        .stApp {
            background-color: #0b1d3a;
        }
        h1, h2, h3, h4, h5, h6 {
            color: #f0f0f0;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center;'>📉 Telecom Churn Prediction Dashboard</h1>", unsafe_allow_html=True)
st.markdown("---")

# --------------------------------
# Load Data & Preprocessing
# --------------------------------
df = pd.read_csv("cleaned_churn_data.csv")
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)
df['Churn'] = df['Churn'].map({'No': 0, 'Yes': 1})
df.drop(['customerID'], axis=1, inplace=True)
if 'Churn_Flag' in df.columns:
    df.drop('Churn_Flag', axis=1, inplace=True)

X = df.drop('Churn', axis=1)
y = df['Churn']
cat_cols = X.select_dtypes(include='object').columns.tolist()
X_enc = pd.get_dummies(X, columns=cat_cols, drop_first=True)

model = RandomForestClassifier(class_weight='balanced', random_state=42)
model.fit(X_enc, y)

# --------------------------------
# User Input
# --------------------------------
st.markdown("### 👤 Enter Customer Details")
col1, col2, col3 = st.columns(3)

inputs = {
    "gender": col1.selectbox("Gender", df["gender"].unique(), key="gender"),
    "SeniorCitizen": col2.selectbox("Senior Citizen", [0, 1], key="senior"),
    "Partner": col3.selectbox("Has Partner?", df["Partner"].unique(), key="partner"),
    "Dependents": col1.selectbox("Has Dependents?", df["Dependents"].unique(), key="dependents"),
    "tenure": col2.slider("Tenure (months)", 0, 80, 12, key="tenure"),
    "PhoneService": col3.selectbox("Phone Service", df["PhoneService"].unique(), key="phone"),
    "MultipleLines": col1.selectbox("Multiple Lines", df["MultipleLines"].unique(), key="multi"),
    "InternetService": col2.selectbox("Internet Service", df["InternetService"].unique(), key="internet"),
    "OnlineSecurity": col3.selectbox("Online Security", df["OnlineSecurity"].unique(), key="security"),
    "OnlineBackup": col1.selectbox("Online Backup", df["OnlineBackup"].unique(), key="backup"),
    "DeviceProtection": col2.selectbox("Device Protection", df["DeviceProtection"].unique(), key="device"),
    "TechSupport": col3.selectbox("Tech Support", df["TechSupport"].unique(), key="support"),
    "StreamingTV": col1.selectbox("Streaming TV", df["StreamingTV"].unique(), key="tv"),
    "StreamingMovies": col2.selectbox("Streaming Movies", df["StreamingMovies"].unique(), key="movies"),
    "Contract": col3.selectbox("Contract Type", df["Contract"].unique(), key="contract"),
    "PaperlessBilling": col1.selectbox("Paperless Billing?", df["PaperlessBilling"].unique(), key="paperless"),
    "PaymentMethod": col2.selectbox("Payment Method", df["PaymentMethod"].unique(), key="payment"),
    "MonthlyCharges": col3.slider("Monthly Charges", 0, 150, 70, key="monthly"),
    "TotalCharges": col1.slider("Total Charges", 0, 10000, 2000, key="total"),
}

# --------------------------------
# Prepare input and predict
# --------------------------------
input_df = pd.DataFrame([inputs])
input_encoded = pd.get_dummies(input_df, drop_first=True)
input_encoded = input_encoded.reindex(columns=X_enc.columns, fill_value=0)

churn_prob = model.predict_proba(input_encoded)[0][1]
churn_class = model.predict(input_encoded)[0]

st.subheader("🎯 Prediction Outcome")
st.markdown("### 🔍 Input Summary")
st.json(inputs)

if churn_class == 1:
    st.error("❌ High Risk of Churn")
else:
    st.success("✅ Low Risk of Churn")

# Custom HTML-based probability bar
st.markdown(f"### 📈 Churn Probability: `{churn_prob:.2f}`")

bar_color = "#e74c3c" if churn_class == 1 else "#3498db"  # red for high churn, blue for low churn
percent = int(churn_prob * 100)

st.markdown(f"""
<div style="background-color: #ddd; border-radius: 5px; height: 25px; width: 100%;">
  <div style="background-color: {bar_color}; width: {percent}%; height: 100%; border-radius: 5px; text-align: center; color: white; font-weight: bold;">
    {percent}%
  </div>
</div>
""", unsafe_allow_html=True)


# --------------------------------
# SHAP Explainability
# --------------------------------
st.markdown("### 🧠 Model Explainability (SHAP)")

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(input_encoded)

if isinstance(shap_values, list):
    shap_array = shap_values[1]  # class 1
else:
    shap_array = shap_values

sample_shap = shap_array[0]
if len(sample_shap.shape) == 2 and sample_shap.shape[1] == 2:
    sample_shap = sample_shap[:, 1]

impact_series = pd.Series(sample_shap, index=input_encoded.columns).sort_values(key=abs, ascending=False)
top_features = impact_series.head(5)

st.write("Top factors influencing this prediction:")
for i, (feature, value) in enumerate(top_features.items(), 1):
    direction = "increased" if value > 0 else "reduced"
    st.markdown(f"{i}. **{feature}** — {direction} churn risk (impact score: `{value:.4f}`)")

st.markdown("""
🔍 **Explanation:**
These features had the strongest influence on this customer's churn prediction.  
Positive values increased churn probability, while negative ones reduced it.

💡 **Suggestions:**
- Provide support or loyalty offers to high-risk customers with low tenure.
- Enable features like `OnlineSecurity`, `TechSupport`, or switch to long-term contracts.
- Encourage paperless billing and automatic payments to reduce churn.
""")
