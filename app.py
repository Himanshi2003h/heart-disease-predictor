import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os

st.set_page_config(
    page_title="Heart Disease Risk Predictor",
    page_icon="🫀",
    layout="centered"
)

@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "heart_disease_model.pkl")
    return joblib.load(model_path)

model = load_model()

st.title("🫀 Heart Disease Risk Predictor")
st.markdown(
    "Enter patient details below to assess heart disease risk. "
    "This tool uses a **Logistic Regression** model trained on 918 patients with **87.5% accuracy** and **91% cross-validation score**."
)
st.divider()

st.subheader("Patient Information")
col1, col2 = st.columns(2)

with col1:
    Age = st.number_input("Age (years)", min_value=18, max_value=100, value=50)
    Sex = st.selectbox("Sex", ["M", "F"])
    ChestPainType = st.selectbox("Chest Pain Type", ["ATA", "NAP", "ASY", "TA"])
    RestingBP = st.number_input("Resting Blood Pressure (mm Hg)", min_value=0, max_value=250, value=120)
    Cholesterol = st.number_input("Cholesterol (mg/dl)", min_value=0, max_value=700, value=200)
    FastingBS = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

with col2:
    RestingECG = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
    MaxHR = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=220, value=150)
    ExerciseAngina = st.selectbox("Exercise Induced Angina", ["N", "Y"])
    Oldpeak = st.number_input("Oldpeak (ST depression)", min_value=-3.0, max_value=7.0, value=0.0, step=0.1)
    ST_Slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

st.divider()

row = pd.DataFrame([{
    'Age': Age,
    'Sex': 1 if Sex == 'M' else 0,
    'ChestPainType': ['ATA', 'NAP', 'ASY', 'TA'].index(ChestPainType),
    'Cholesterol': Cholesterol,
    'FastingBS': FastingBS,
    'MaxHR': MaxHR,
    'ExerciseAngina': 1 if ExerciseAngina == 'Y' else 0,
    'Oldpeak': Oldpeak,
    'ST_Slope': ['Down', 'Flat', 'Up'].index(ST_Slope),
}])

row['Oldpeak'] = (row['Oldpeak'] - (-2.6)) / (6.2 - (-2.6))
row['Age'] = (row['Age'] - 53.5) / 9.4
row['Cholesterol'] = (row['Cholesterol'] - 198.8) / 109.4
row['MaxHR'] = (row['MaxHR'] - 136.8) / 25.5

features = row.values

if st.button("Predict Risk", type="primary", use_container_width=True):
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0]
    risk_pct = round(probability[1] * 100, 1)

    st.divider()
    st.subheader("Prediction Result")

    if prediction == 1:
        st.error(f"⚠️ High Risk — {risk_pct}% probability of heart disease")
        st.markdown("The model indicates a **high likelihood** of heart disease. Please consult a qualified medical professional.")
    else:
        st.success(f"✅ Low Risk — {risk_pct}% probability of heart disease")
        st.markdown("The model indicates a **low likelihood** of heart disease. Continue maintaining a healthy lifestyle.")

    col_a, col_b = st.columns(2)
    col_a.metric("Low Risk", f"{round(probability[0]*100, 1)}%")
    col_b.metric("High Risk", f"{round(probability[1]*100, 1)}%")
    st.progress(int(risk_pct))

st.divider()
st.caption("Built by Himanshi Surage · M.Tech CSE, NIT Warangal · Logistic Regression · Accuracy: 87.5% · CV Score: 91% · Dataset: Heart Failure Prediction (Kaggle, 918 patients)")
