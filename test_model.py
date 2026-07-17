import streamlit as st
import joblib

st.set_page_config(page_title="Credit Default Prediction", layout="wide")

@st.cache_resource
def load_model():
    model = joblib.load("credit_default_model.pkl")
    scaler = joblib.load("scaler(1).pkl")
    return model, scaler

model, scaler = load_model()

st.title("💳 Credit Default Prediction System")
st.success("✅ Model Loaded Successfully")