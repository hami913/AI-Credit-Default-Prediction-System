from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & CUSTOM STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Credit Risk AI Engine",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-End Fintech UI/UX CSS Injection
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

    /* Global Typography & Background */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }
    
    .stApp {
        background: radial-gradient(circle at 50% -20%, #1e1b4b 0%, #0f172a 60%, #020617 100%);
        color: #f8fafc;
    }

    /* Top Navigation / Hero Header */
    .hero-container {
        padding: 1.5rem 0 1rem 0;
        margin-bottom: 1.5rem;
    }
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        border-radius: 9999px;
        background: rgba(56, 189, 248, 0.1);
        border: 1px solid rgba(56, 189, 248, 0.3);
        color: #38bdf8;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 0.8rem;
    }
    .hero-title {
        font-size: 2.5rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        background: linear-gradient(135deg, #ffffff 0%, #cbd5e1 50%, #38bdf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        line-height: 1.2;
    }
    .hero-subtitle {
        color: #94a3b8;
        font-size: 1rem;
        font-weight: 400;
        margin-top: 0.4rem;
    }

    /* Glassmorphism Input Cards */
    .glass-card {
        background: rgba(30, 41, 59, 0.5);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        padding: 24px;
        box-shadow: 0 20px 40px -15px rgba(0, 0, 0, 0.5);
        margin-bottom: 1.5rem;
    }
    .card-header {
        font-size: 1.15rem;
        font-weight: 700;
        color: #f8fafc;
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 1.2rem;
        padding-bottom: 10px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    }

    /* Field Captions Styling */
    .field-caption {
        background: rgba(15, 23, 42, 0.6);
        border-left: 3px solid #6366f1;
        padding: 8px 12px;
        border-radius: 0 8px 8px 0;
        color: #cbd5e1;
        font-size: 0.8rem;
        margin-top: -8px;
        margin-bottom: 16px;
        line-height: 1.4;
    }

    /* Streamlit Input Overrides */
    div[data-baseweb="input"] {
        background-color: rgba(15, 23, 42, 0.8) !important;
        border-color: rgba(255, 255, 255, 0.1) !important;
        border-radius: 10px !important;
        color: white !important;
    }
    div[data-baseweb="select"] > div {
        background-color: rgba(15, 23, 42, 0.8) !important;
        border-color: rgba(255, 255, 255, 0.1) !important;
        border-radius: 10px !important;
        color: white !important;
    }
    label {
        font-weight: 600 !important;
        color: #e2e8f0 !important;
        font-size: 0.9rem !important;
    }

    /* ULTRA-MODERN NEON BUTTON */
    div.stButton > button {
        width: 100% !important;
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 50%, #d946ef 100%) !important;
        color: #ffffff !important;
        border: none !important;
        font-weight: 700 !important;
        font-size: 1.1rem !important;
        border-radius: 14px !important;
        padding: 18px 32px !important;
        box-shadow: 0 4px 25px rgba(124, 58, 237, 0.4) !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        letter-spacing: 0.03em !important;
        cursor: pointer !important;
    }
    div.stButton > button:hover {
        transform: translateY(-3px) scale(1.005) !important;
        box-shadow: 0 10px 35px rgba(124, 58, 237, 0.6) !important;
        background: linear-gradient(135deg, #4338ca 0%, #6d28d9 50%, #c026d3 100%) !important;
    }
    div.stButton > button:active {
        transform: translateY(1px) !important;
    }

    /* Decision Banner Styling */
    .decision-card-high {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.15) 0%, rgba(153, 27, 27, 0.2) 100%);
        border: 1px solid rgba(239, 68, 68, 0.4);
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 20px;
    }
    .decision-card-mod {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.15) 0%, rgba(180, 83, 9, 0.2) 100%);
        border: 1px solid rgba(245, 158, 11, 0.4);
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 20px;
    }
    .decision-card-low {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(6, 95, 70, 0.2) 100%);
        border: 1px solid rgba(16, 185, 129, 0.4);
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 20px;
    }
    .decision-title {
        font-size: 1.25rem;
        font-weight: 800;
        margin-bottom: 6px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .decision-desc {
        color: #cbd5e1;
        font-size: 0.9rem;
        margin: 0;
    }

    /* Risk Factor Pills */
    .factor-item {
        background: rgba(15, 23, 42, 0.7);
        border: 1px solid rgba(239, 68, 68, 0.25);
        border-radius: 10px;
        padding: 12px 16px;
        margin-bottom: 10px;
        color: #fca5a5;
        font-size: 0.88rem;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .factor-item-safe {
        background: rgba(15, 23, 42, 0.7);
        border: 1px solid rgba(16, 185, 129, 0.25);
        border-radius: 10px;
        padding: 12px 16px;
        color: #6ee7b7;
        font-size: 0.88rem;
        display: flex;
        align-items: center;
        gap: 10px;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 2. MODEL ARTIFACT LOADING
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "xgboost_model.pkl"
SCALER_PATH = BASE_DIR / "models" / "scaler.pkl"

@st.cache_resource
def load_artifacts():
    if not MODEL_PATH.exists() or not SCALER_PATH.exists():
        st.error("⚠️ Model or Scaler artifact missing! Please run 'python train_model.py' first.")
        st.stop()
        
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

model, scaler = load_artifacts()


# -----------------------------------------------------------------------------
# 3. HERO HEADER
# -----------------------------------------------------------------------------
st.markdown("""
<div class="hero-container">
    <div class="status-badge">
        <span style="height: 6px; width: 6px; background-color: #38bdf8; border-radius: 50%; display: inline-block;"></span>
        AI-Powered Risk Analytics v2.0
    </div>
    <div class="hero-title">Credit Default Assessment Engine</div>
    <div class="hero-subtitle">Enter applicant financial metrics to generate real-time machine learning risk scores and explainability insights.</div>
</div>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 4. INPUT DASHBOARD FORM
# -----------------------------------------------------------------------------
col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown('<div class="glass-card"><div class="card-header">👤 Applicant Demographics & Score</div>', unsafe_allow_html=True)
    
    age = st.number_input("Age (Years)", min_value=18, max_value=100, value=30, step=1)
    st.markdown('<div class="field-caption">💡 <b>Age:</b> Applicant ki umar years me — financial stability aur career stage estimate karti hai.</div>', unsafe_allow_html=True)
    
    income = st.number_input("Annual Income ($)", min_value=0, max_value=2000000, value=65000, step=1000)
    st.markdown('<div class="field-caption">💡 <b>Annual Income:</b> Applicant ki kul saalana aamdani jo loan wapas karne ki capacity batati hai.</div>', unsafe_allow_html=True)
    
    credit_score = st.number_input("Credit Score (FICO)", min_value=300, max_value=850, value=710, step=1)
    st.markdown('<div class="field-caption">💡 <b>Credit Score:</b> Borrower ki past loan repayment history aur trustworthiness ka 300-850 score.</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="glass-card"><div class="card-header">💰 Loan & Obligations Profile</div>', unsafe_allow_html=True)
    
    loan_amount = st.number_input("Loan Amount Requested ($)", min_value=0, max_value=1000000, value=15000, step=500)
    st.markdown('<div class="field-caption">💡 <b>Loan Amount:</b> Kitan qarza maanga gaya hai. Bada loan ziada risk paida karta hai.</div>', unsafe_allow_html=True)
    
    debt_to_income = st.number_input("Debt-to-Income (DTI) Ratio", min_value=0.0, max_value=1.0, value=0.25, step=0.01)
    st.markdown('<div class="field-caption">💡 <b>DTI Ratio:</b> Monthly debts ka income se ratio. High ratio matlab banda pehle hi qarzay me daba hua hai.</div>', unsafe_allow_html=True)
    
    delinquencies = st.selectbox("Past Delinquencies", options=[0, 1, 2, 3, 4, 5], index=0)
    st.markdown('<div class="field-caption">💡 <b>Past Delinquencies:</b> Pehle miss ya late ki gayi monthly payments ki tadad.</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

st.write("")

# -----------------------------------------------------------------------------
# 5. EXECUTION & PREDICTION
# -----------------------------------------------------------------------------
assess_btn = st.button("⚡ EXECUTE RISK ASSESSMENT", type="primary")

if assess_btn:
    # DataFrame construction
    input_df = pd.DataFrame([{
        'age': age,
        'income': income,
        'loan_amount': loan_amount,
        'credit_score': credit_score,
        'debt_to_income': debt_to_income,
        'delinquencies': delinquencies
    }])

    # Scale & Predict
    scaled_features = scaler.transform(input_df)
    default_prob = float(model.predict_proba(scaled_features)[0][1])
    risk_percentage = default_prob * 100

    st.markdown("---")
    
    st.markdown('<div style="font-size: 1.5rem; font-weight: 800; margin-bottom: 1rem; color: #f8fafc;">📊 Assessment Results & Explainability</div>', unsafe_allow_html=True)

    res_col1, res_col2 = st.columns([1, 1], gap="large")

    # Column 1: Vibrant Multi-Color Spectrum Gauge Chart
    with res_col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        
        # Dynamic theme attributes based on calculated risk severity
        if default_prob >= 0.50:
            gauge_color = "#f43f5e"      # Neon Rose / Red
            status_text = "CRITICAL RISK"
            status_bg = "rgba(244, 63, 94, 0.15)"
            status_border = "#f43f5e"
        elif default_prob >= 0.25:
            gauge_color = "#f59e0b"      # Electric Amber
            status_text = "MODERATE RISK"
            status_bg = "rgba(245, 158, 11, 0.15)"
            status_border = "#f59e0b"
        else:
            gauge_color = "#10b981"      # Emerald Green
            status_text = "LOW DEFAULT RISK"
            status_bg = "rgba(16, 185, 129, 0.15)"
            status_border = "#10b981"
        
        # Plotly Colorful Gauge Chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_percentage,
            number={
                'suffix': "%", 
                'font': {'size': 48, 'family': 'Plus Jakarta Sans', 'color': gauge_color, 'weight': 800}
            },
            title={
                'text': "<span style='letter-spacing: 1.5px;'>DEFAULT RISK METER</span>", 
                'font': {'size': 13, 'family': 'Plus Jakarta Sans', 'color': '#94a3b8', 'weight': 700}
            },
            gauge={
                'axis': {
                    'range': [0, 100], 
                    'tickwidth': 2, 
                    'tickcolor': "#64748b", 
                    'dtick': 20,
                    'tickfont': {'size': 11, 'color': '#cbd5e1', 'family': 'Plus Jakarta Sans'}
                },
                'bar': {'color': gauge_color, 'thickness': 0.28},
                'bgcolor': "rgba(15, 23, 42, 0.8)",
                'borderwidth': 2,
                'bordercolor': "rgba(255, 255, 255, 0.12)",
                # VIBRANT 5-STAGE COLOR SPECTRUM TRACK
                'steps': [
                    {'range': [0, 15], 'color': '#10b981'},     # Vibrant Emerald
                    {'range': [15, 30], 'color': '#06b6d4'},    # Bright Cyan
                    {'range': [30, 50], 'color': '#eab308'},    # Electric Yellow
                    {'range': [50, 75], 'color': '#f97316'},    # Vivid Orange
                    {'range': [75, 100], 'color': '#f43f5e'}    # Neon Red
                ],
                'threshold': {
                    'line': {'color': "#ffffff", 'width': 4},
                    'thickness': 0.85,
                    'value': risk_percentage
                }
            }
        ))
        
        fig.update_layout(
            height=300, 
            margin=dict(l=30, r=30, t=50, b=10),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Glowing Status Badge directly under the meter
        st.markdown(f"""
        <div style="text-align: center; margin-top: -10px; margin-bottom: 10px;">
            <span style="background: {status_bg}; border: 1px solid {status_border}; color: {gauge_color}; font-weight: 800; font-size: 0.85rem; padding: 6px 18px; border-radius: 9999px; letter-spacing: 1px; display: inline-flex; align-items: center; gap: 6px;">
                ⚡ {status_text} • {risk_percentage:.1f}%
            </span>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

    # Column 2: Decision Banner & Explainability List
    with res_col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        
        # Risk Decision Cards
        if default_prob >= 0.50:
            st.markdown("""
            <div class="decision-card-high">
                <div class="decision-title" style="color: #fca5a5;">🚨 HIGH RISK / REJECTED</div>
                <div class="decision-desc">Applicant exceeds default risk tolerance thresholds. Loan approval is not recommended.</div>
            </div>
            """, unsafe_allow_html=True)
        elif default_prob >= 0.25:
            st.markdown("""
            <div class="decision-card-mod">
                <div class="decision-title" style="color: #fde68a;">⚠️ MODERATE RISK / MANUAL REVIEW</div>
                <div class="decision-desc">Applicant profile carries conditional risk. Secondary underwriting evaluation is recommended.</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="decision-card-low">
                <div class="decision-title" style="color: #6ee7b7;">✅ LOW RISK / APPROVED</div>
                <div class="decision-desc">Applicant profile aligns with low-risk acceptance guidelines. Eligible for standard processing.</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div style="font-size: 1rem; font-weight: 700; color: #f8fafc; margin-bottom: 12px;">🔍 Risk Factors Breakdown (Default Ki Wajha)</div>', unsafe_allow_html=True)
        
        # Explainability Logic
        risk_factors = []
        
        if credit_score < 620:
            risk_factors.append(f"<b>Low Credit Score ({credit_score}):</b> Score 620 se kam hone ki wajha se default risk ziada hai.")
        if debt_to_income > 0.35:
            risk_factors.append(f"<b>High DTI Ratio ({debt_to_income:.2f}):</b> Monthly income ka bada hissa qarzay chukane me ja raha hai.")
        if delinquencies > 0:
            risk_factors.append(f"<b>Past Delinquencies ({delinquencies}):</b> Pehle ki gayi payments late/miss karne ka record hai.")
        if income > 0 and (loan_amount / income) > 0.4:
            risk_factors.append(f"<b>High Loan-to-Income Ratio:</b> Requested loan amount saalana aamdani ke muqable me kafi zyada hai.")
            
        if risk_factors:
            for factor in risk_factors:
                st.markdown(f'<div class="factor-item"><span>🔴</span> <div>{factor}</div></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="factor-item-safe"><span>🟢</span> <div><b>Strong Financial Health:</b> Credit Score acha hai, DTI low hai, aur koi past delinquency record nahi hai.</div></div>', unsafe_allow_html=True)
            
        st.markdown('</div>', unsafe_allow_html=True)