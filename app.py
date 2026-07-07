import streamlit as st
import joblib
import xgboost
from xgboost import XGBClassifier
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# =========================
# CONFIG
# =========================

st.set_page_config(
    page_title="AI Credit Default Prediction",
    page_icon="💳",
    layout="wide"
)

MODEL_PATH = "credit_default_model(1).json"
SCALER_PATH = "scaler(4).pkl"

COLOR_SAFE = "#00E676"
COLOR_DEFAULT = "#FF3D57"
COLOR_ACCENT = "#00E5FF"
COLOR_BG = "#0E1117"

EDU_MAP = {1: "Graduate School", 2: "University", 3: "High School", 4: "Others"}
MAR_MAP = {1: "Married", 2: "Single", 3: "Others"}
PAY_STATUS_MAP = {
    -2: "No Consumption", -1: "Paid Duly", 0: "Use of Revolving Credit",
    1: "1 Month Delay", 2: "2 Months Delay", 3: "3 Months Delay",
    4: "4 Months Delay", 5: "5 Months Delay", 6: "6 Months Delay",
    7: "7 Months Delay", 8: "8+ Months Delay"
}
STATUS_OPTIONS = list(PAY_STATUS_MAP.keys())

FEATURE_ORDER = [
    "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
    "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
]

FEATURE_LABELS = {
    "LIMIT_BAL": "Credit Limit", "SEX": "Gender", "EDUCATION": "Education", "MARRIAGE": "Marital Status",
    "AGE": "Age", "PAY_0": "Sept. Payment Status", "PAY_2": "Aug. Payment Status",
    "PAY_3": "July Payment Status", "PAY_4": "June Payment Status", "PAY_5": "May Payment Status",
    "PAY_6": "April Payment Status",
    "BILL_AMT1": "Sept. Bill", "BILL_AMT2": "Aug. Bill", "BILL_AMT3": "July Bill",
    "BILL_AMT4": "June Bill", "BILL_AMT5": "May Bill", "BILL_AMT6": "April Bill",
    "PAY_AMT1": "Sept. Payment", "PAY_AMT2": "Aug. Payment", "PAY_AMT3": "July Payment",
    "PAY_AMT4": "June Payment", "PAY_AMT5": "May Payment", "PAY_AMT6": "April Payment",
}


# =========================
# MODEL LOADING
# =========================

@st.cache_resource
def load_model():
    model = XGBClassifier()
    model.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler


@st.cache_resource
def get_shap_explainer(_model):
    if not SHAP_AVAILABLE:
        return None
    try:
        return shap.TreeExplainer(_model)
    except Exception:
        return None


def predict_default(model, scaler, inputs: dict) -> tuple[float, np.ndarray]:
    """Returns (risk percentage, scaled feature row) for reuse in explainability."""
    row = pd.DataFrame([[inputs[col] for col in FEATURE_ORDER]], columns=FEATURE_ORDER)
    scaled = scaler.transform(row)
    proba = model.predict_proba(scaled)[0]
    return float(proba[1]) * 100, scaled


def local_explanation(model, scaled_row):
    """Returns a DataFrame of top feature contributions for this single prediction."""
    explainer = get_shap_explainer(model)
    if explainer is not None:
        shap_values = explainer.shap_values(scaled_row)
        values = shap_values[0] if isinstance(shap_values, list) else shap_values[0]
        df = pd.DataFrame({
            "Feature": [FEATURE_LABELS[f] for f in FEATURE_ORDER],
            "Impact": values
        })
        df["Direction"] = np.where(df["Impact"] >= 0, "Increases Risk", "Decreases Risk")
        return df.reindex(df["Impact"].abs().sort_values(ascending=False).index).head(8), True
    # Fallback: global feature_importances_ scaled by how extreme this customer's inputs are
    importances = model.feature_importances_
    z = (scaled_row[0] - scaled_row[0].mean())
    approx_impact = importances * z
    df = pd.DataFrame({
        "Feature": [FEATURE_LABELS[f] for f in FEATURE_ORDER],
        "Impact": approx_impact
    })
    df["Direction"] = np.where(df["Impact"] >= 0, "Increases Risk", "Decreases Risk")
    return df.reindex(df["Impact"].abs().sort_values(ascending=False).index).head(8), False


def generate_ai_narrative(risk, top_factors_df):
    """Plain-language, analyst-style summary of the prediction — the 'AI agent' voice."""
    band = "low" if risk < 30 else "medium" if risk < 70 else "high"
    lead = {
        "low": "This customer presents a low likelihood of default based on their current profile.",
        "medium": "This customer shows a moderate default risk that warrants a closer look before approval.",
        "high": "This customer's profile carries a high probability of default and should be reviewed carefully."
    }[band]

    drivers = top_factors_df.head(3)
    driver_sentences = []
    for _, row in drivers.iterrows():
        verb = "pushing risk up" if row["Direction"] == "Increases Risk" else "helping keep risk down"
        driver_sentences.append(f"**{row['Feature']}** ({verb})")

    driver_text = ", ".join(driver_sentences)
    return f"{lead} The factors with the largest influence on this result are {driver_text}."


# =========================
# STYLING
# =========================

def inject_css():
    st.markdown(f"""
    <style>
    .main {{ background-color: {COLOR_BG}; }}
    .big-title {{ font-size: 40px; font-weight: bold; color: white; }}
    .sub-title {{ color: #B8BCC8; font-size: 18px; }}
    .ai-narrative {{
        background: linear-gradient(135deg, #1E1E2F, #16213E);
        border-left: 4px solid {COLOR_ACCENT};
        padding: 18px 22px;
        border-radius: 10px;
        color: #E8E8E8;
        font-size: 16px;
        line-height: 1.6;
    }}
    div.stButton > button {{
        width: 100%;
        background: linear-gradient(90deg, {COLOR_ACCENT}, {COLOR_SAFE});
        color: #0E1117;
        font-weight: 700;
        font-size: 18px;
        padding: 0.6em 0;
        border-radius: 12px;
        border: none;
        transition: transform 0.15s ease;
    }}
    div.stButton > button:hover {{ transform: scale(1.02); }}
    </style>
    """, unsafe_allow_html=True)


# =========================
# SHARED CHART BUILDERS
# =========================

def build_pie(risk):
    pie_data = pd.DataFrame({"Category": ["🟢 Safe", "🔴 Default"], "Probability": [100 - risk, risk]})
    pie = px.pie(
        pie_data, names="Category", values="Probability", hole=0.65, color="Category",
        color_discrete_map={"🟢 Safe": COLOR_SAFE, "🔴 Default": COLOR_DEFAULT}
    )
    pie.update_traces(textinfo="percent+label", textfont_size=16, pull=[0, 0.08],
                       marker=dict(line=dict(color="white", width=3)))
    pie.update_layout(
        title="🥧 Risk Distribution", paper_bgcolor=COLOR_BG, plot_bgcolor=COLOR_BG, font=dict(color="white"),
        annotations=[dict(text=f"<b>{risk:.1f}%</b><br>Risk", x=0.5, y=0.5, showarrow=False, font=dict(size=22))]
    )
    return pie


def build_gauge(risk):
    gauge = go.Figure(go.Indicator(
        mode="gauge+number", value=risk, number={"suffix": "%"},
        gauge={
            "axis": {"range": [0, 100]}, "bar": {"color": COLOR_ACCENT},
            "steps": [
                {"range": [0, 30], "color": COLOR_SAFE},
                {"range": [30, 70], "color": "#FFD54F"},
                {"range": [70, 100], "color": COLOR_DEFAULT},
            ],
            "threshold": {"line": {"color": "white", "width": 4}, "value": risk, "thickness": 0.8},
        }
    ))
    gauge.update_layout(title="🎯 Credit Risk Meter", paper_bgcolor=COLOR_BG, font=dict(color="white"), height=350)
    return gauge


def build_explanation_chart(exp_df):
    exp_df = exp_df.iloc[::-1]  # largest impact at top when horizontal
    fig = px.bar(
        exp_df, x="Impact", y="Feature", orientation="h", color="Direction",
        color_discrete_map={"Increases Risk": COLOR_DEFAULT, "Decreases Risk": COLOR_SAFE}
    )
    fig.update_layout(
        title="🧠 Why the AI Made This Decision", paper_bgcolor=COLOR_BG, plot_bgcolor=COLOR_BG,
        font=dict(color="white"), showlegend=True, height=400
    )
    return fig


# =========================
# TAB 1: SINGLE PREDICTION
# =========================

def render_single_prediction_tab():
    st.subheader("👤 Customer Information")
    col1, col2, col3 = st.columns(3)
    with col1:
        limit_bal = st.number_input("Credit Limit (LIMIT_BAL)", min_value=10000, max_value=1_000_000, value=50000, step=1000)
    with col2:
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
    with col3:
        sex = st.selectbox("Gender", [1, 2], format_func=lambda x: "Male" if x == 1 else "Female")

    col4, col5 = st.columns(2)
    with col4:
        education = st.selectbox("Education", list(EDU_MAP.keys()), format_func=lambda x: EDU_MAP[x])
    with col5:
        marriage = st.selectbox("Marriage", list(MAR_MAP.keys()), format_func=lambda x: MAR_MAP[x])

    st.markdown("---")
    st.subheader("📅 Payment History")
    labels = [("PAY_0", "Sept. (PAY_0)"), ("PAY_2", "Aug. (PAY_2)"), ("PAY_3", "July (PAY_3)"),
              ("PAY_4", "June (PAY_4)"), ("PAY_5", "May (PAY_5)"), ("PAY_6", "April (PAY_6)")]
    payments = {}
    for row_labels in (labels[:3], labels[3:]):
        cols = st.columns(3)
        for col, (key, label) in zip(cols, row_labels):
            with col:
                payments[key] = st.selectbox(label, STATUS_OPTIONS, format_func=lambda x: PAY_STATUS_MAP[x], key=key)

    st.markdown("---")
    st.subheader("💳 Bill & Payment Amounts")
    months = ["September", "August", "July", "June", "May", "April"]
    bill_defaults = [5000, 4500, 4000, 3500, 3000, 2500]
    pay_defaults = [2000, 1800, 1600, 1400, 1200, 1000]
    bills, pay_amts = {}, {}

    with st.expander("Bill Amounts (BILL_AMT1–6)", expanded=True):
        for i in range(0, 6, 3):
            cols = st.columns(3)
            for j, col in enumerate(cols):
                idx = i + j
                key = f"BILL_AMT{idx + 1}"
                with col:
                    bills[key] = st.number_input(f"{months[idx]}", min_value=0, value=bill_defaults[idx], step=100, key=key)

    with st.expander("Payment Amounts (PAY_AMT1–6)", expanded=True):
        for i in range(0, 6, 3):
            cols = st.columns(3)
            for j, col in enumerate(cols):
                idx = i + j
                key = f"PAY_AMT{idx + 1}"
                with col:
                    pay_amts[key] = st.number_input(f"{months[idx]}", min_value=0, value=pay_defaults[idx], step=100, key=key)

    st.markdown("---")
    if st.button("🔮 Predict Credit Risk", key="single_predict"):
        inputs = {"LIMIT_BAL": limit_bal, "AGE": age, "SEX": sex, "EDUCATION": education,
                   "MARRIAGE": marriage, **payments, **bills, **pay_amts}
        try:
            model, scaler = load_model()
            risk, scaled_row = predict_default(model, scaler, inputs)
            confidence = max(risk, 100 - risk)
            exp_df, is_shap = local_explanation(model, scaled_row)

            # Log to session history
            st.session_state.setdefault("history", [])
            st.session_state["history"].append({
                "Age": age, "Credit Limit": limit_bal, "Gender": "Male" if sex == 1 else "Female",
                "Risk %": round(risk, 2),
                "Band": "Low" if risk < 30 else "Medium" if risk < 70 else "High"
            })

            render_results(risk, confidence, exp_df, is_shap, age, limit_bal, sex, education, marriage)
        except FileNotFoundError as e:
            st.error(f"⚠️ Model or scaler file not found: {e}. Check that '{MODEL_PATH}' and '{SCALER_PATH}' exist.")
    else:
        st.info("👆 Fill in the details above and click **Predict Credit Risk** to see results.")


def render_results(risk, confidence, exp_df, is_shap, age, limit_bal, sex, education, marriage):
    st.markdown("---")
    st.subheader("📊 Prediction Dashboard")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🔴 Default Risk", f"{risk:.2f}%")
    with col2:
        st.metric("🟢 Safe Probability", f"{100 - risk:.2f}%")
    with col3:
        st.metric("🤖 Model Confidence", f"{confidence:.2f}%")

    # AI narrative — the "agent" voice
    narrative = generate_ai_narrative(risk, exp_df)
    st.markdown(f"<div class='ai-narrative'>🤖 <b>AI Analyst Summary</b><br><br>{narrative}</div>", unsafe_allow_html=True)
    st.write("")

    left, right = st.columns(2)
    with left:
        st.plotly_chart(build_pie(risk), use_container_width=True)
    with right:
        st.plotly_chart(build_gauge(risk), use_container_width=True)

    st.markdown("---")
    st.plotly_chart(build_explanation_chart(exp_df), use_container_width=True)
    if not is_shap:
        st.caption("ℹ️ Install the `shap` package for exact per-customer explanations. "
                   "Showing an approximation based on global feature importance for now.")

    st.markdown("---")
    st.subheader("📈 Overall Credit Risk")
    st.progress(int(risk))
    st.write(f"Current Credit Risk: **{risk:.2f}%**")

    st.markdown("---")
    st.subheader("🤖 AI Recommendation")
    if risk < 30:
        st.success("✅ Customer has a low probability of default. Credit approval is recommended.")
    elif risk < 70:
        st.warning("⚠️ Medium credit risk detected. Manual verification is recommended before approval.")
    else:
        st.error("❌ High probability of default. Credit approval is not recommended without further review.")

    st.markdown("---")
    st.subheader("📥 Download Prediction Report")
    report = pd.DataFrame({
        "Feature": ["Age", "Credit Limit", "Gender", "Education", "Marriage", "Risk Level", "Default Probability"],
        "Value": [age, limit_bal, "Male" if sex == 1 else "Female", EDU_MAP[education], MAR_MAP[marriage],
                  "Low Risk" if risk < 30 else "Medium Risk" if risk < 70 else "High Risk", f"{risk:.2f}%"]
    })
    csv = report.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download CSV Report", data=csv, file_name="credit_prediction_report.csv",
                        mime="text/csv", use_container_width=True)


# =========================
# TAB 2: BATCH PREDICTION
# =========================

def render_batch_tab():
    st.subheader("📂 Batch Prediction")
    st.write("Upload a CSV with columns matching the model's expected features to score many customers at once.")
    st.code(", ".join(FEATURE_ORDER), language="text")

    uploaded = st.file_uploader("Upload customer CSV", type=["csv"])
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            missing = [c for c in FEATURE_ORDER if c not in df.columns]
            if missing:
                st.error(f"⚠️ Missing required columns: {', '.join(missing)}")
                return

            model, scaler = load_model()
            scaled = scaler.transform(df[FEATURE_ORDER])
            proba = model.predict_proba(scaled)[:, 1] * 100

            results = df.copy()
            results["Default Risk %"] = proba.round(2)
            results["Risk Band"] = pd.cut(proba, bins=[-1, 30, 70, 101], labels=["Low", "Medium", "High"])

            st.success(f"✅ Scored {len(results)} customers.")
            st.dataframe(results, use_container_width=True)

            fig = px.histogram(results, x="Default Risk %", color="Risk Band", nbins=20,
                                color_discrete_map={"Low": COLOR_SAFE, "Medium": "#FFD54F", "High": COLOR_DEFAULT})
            fig.update_layout(paper_bgcolor=COLOR_BG, plot_bgcolor=COLOR_BG, font=dict(color="white"),
                               title="Portfolio Risk Distribution")
            st.plotly_chart(fig, use_container_width=True)

            csv = results.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Scored Results", data=csv, file_name="batch_predictions.csv",
                                mime="text/csv", use_container_width=True)
        except FileNotFoundError as e:
            st.error(f"⚠️ Model or scaler file not found: {e}")
        except Exception as e:
            st.error(f"⚠️ Could not process file: {e}")


# =========================
# TAB 3: MODEL INSIGHTS
# =========================

def render_insights_tab():
    st.subheader("📈 Model Insights")
    try:
        model, _ = load_model()
        importances = model.feature_importances_
        imp_df = pd.DataFrame({
            "Feature": [FEATURE_LABELS[f] for f in FEATURE_ORDER],
            "Importance": importances
        }).sort_values("Importance", ascending=True)

        fig = px.bar(imp_df, x="Importance", y="Feature", orientation="h",
                     color="Importance", color_continuous_scale=["#1E1E2F", COLOR_ACCENT])
        fig.update_layout(paper_bgcolor=COLOR_BG, plot_bgcolor=COLOR_BG, font=dict(color="white"),
                           title="Global Feature Importance (what the model relies on most)", height=600)
        st.plotly_chart(fig, use_container_width=True)
    except FileNotFoundError as e:
        st.error(f"⚠️ Model file not found: {e}")

    st.markdown("---")
    st.subheader("🧪 Reported Model Performance")
    st.caption("Populate these with your actual held-out test metrics.")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", "84%")
    m2.metric("Precision", "—")
    m3.metric("Recall", "—")
    m4.metric("ROC-AUC", "—")


# =========================
# TAB 4: HISTORY
# =========================

def render_history_tab():
    st.subheader("🕒 Session Prediction History")
    history = st.session_state.get("history", [])
    if not history:
        st.info("No predictions made yet this session. Run one from the **Single Prediction** tab.")
        return

    hist_df = pd.DataFrame(history)
    st.dataframe(hist_df, use_container_width=True)

    fig = px.line(hist_df.reset_index(), x="index", y="Risk %", markers=True,
                   title="Risk Score Across This Session's Predictions")
    fig.update_layout(paper_bgcolor=COLOR_BG, plot_bgcolor=COLOR_BG, font=dict(color="white"),
                       xaxis_title="Prediction #")
    st.plotly_chart(fig, use_container_width=True)

    if st.button("🗑️ Clear History"):
        st.session_state["history"] = []
        st.rerun()


# =========================
# MAIN
# =========================

def render_header():
    st.markdown("<div class='big-title'>💳 AI Credit Default Prediction System</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='sub-title'>An AI-powered credit risk assistant — score customers, "
        "understand why, and explore your portfolio.</div>", unsafe_allow_html=True
    )
    st.write("")
    c1, c2, c3, c4 = st.columns(4)
    c1.info("🤖 **Model**\n\nXGBoost Classifier")
    c2.success("📊 **Features**\n\n23 Inputs")
    c3.warning("⚡ **Accuracy**\n\n84%")
    c4.info("🧠 **Explainability**\n\n" + ("SHAP Enabled" if SHAP_AVAILABLE else "Approximate Mode"))
    st.markdown("---")


def main():
    inject_css()
    render_header()

    tab1, tab2, tab3, tab4 = st.tabs(["🧍 Single Prediction", "📂 Batch Prediction", "📈 Model Insights", "🕒 History"])
    with tab1:
        render_single_prediction_tab()
    with tab2:
        render_batch_tab()
    with tab3:
        render_insights_tab()
    with tab4:
        render_history_tab()


if __name__ == "__main__":
    main()