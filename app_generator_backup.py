from pathlib import Path

code = r'''from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from src.feature_engineering import add_credit_features


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Credit Risk Intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "credit_default_model_v1.joblib"
REPORT_PATH = BASE_DIR / "reports" / "credit_risk_regulatory_report.json"


# ============================================================
# LOAD ASSETS
# ============================================================

@st.cache_resource
def load_model_artifact():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_governance_report():
    if REPORT_PATH.exists():
        with open(REPORT_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


try:
    artifact = load_model_artifact()
except FileNotFoundError:
    st.error(f"Model file not found: `{MODEL_PATH}`")
    st.stop()
except Exception as exc:
    st.error(f"Could not load model: {exc}")
    st.stop()

report = load_governance_report()

model = artifact["pipeline"]
threshold = float(artifact.get("threshold", 0.50))
model_features = artifact["features"]
model_name = artifact.get("model_name", "Credit Default Random Forest")
model_version = artifact.get("model_version", "1.0")


# ============================================================
# CONSTANTS
# ============================================================

MONTHS = ["September", "August", "July", "June", "May", "April"]

STATUS_MAP = {
    "Special Status (-2)": -2,
    "Special Status (-1)": -1,
    "No Delay": 0,
    "1 Month Delay": 1,
    "2 Months Delay": 2,
    "3 Months Delay": 3,
    "4 Months Delay": 4,
    "5 Months Delay": 5,
    "6 Months Delay": 6,
    "7 Months Delay": 7,
    "8 Months Delay": 8,
}

DEFAULT_HISTORY = pd.DataFrame(
    {
        "Month": MONTHS,
        "Repayment Status": ["No Delay"] * 6,
        "Statement Balance": [0.0] * 6,
        "Payment Made": [0.0] * 6,
    }
)

DEMO_HISTORY = pd.DataFrame(
    {
        "Month": MONTHS,
        "Repayment Status": ["No Delay"] * 6,
        "Statement Balance": [
            50000.0, 45000.0, 40000.0,
            35000.0, 30000.0, 25000.0,
        ],
        "Payment Made": [5000.0] * 6,
    }
)


# ============================================================
# STYLE
# ============================================================

st.markdown(
    """
    <style>
        :root {
            --bg-card: rgba(255,255,255,.035);
            --border: rgba(128,128,128,.18);
            --muted: rgba(127,127,127,.78);
            --blue: #4f8cff;
            --green: #24a47c;
            --red: #d95d5d;
        }

        .stApp {
            background:
                radial-gradient(circle at 10% -5%, rgba(79,140,255,.12), transparent 30%),
                radial-gradient(circle at 92% 3%, rgba(36,164,124,.09), transparent 27%),
                linear-gradient(180deg, rgba(255,255,255,.015), transparent 22%);
        }

        .block-container {
            max-width: 1280px;
            padding-top: 1.35rem;
            padding-bottom: 3rem;
        }

        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}

        .hero {
            position: relative;
            overflow: hidden;
            border: 1px solid var(--border);
            border-radius: 24px;
            padding: 34px 36px;
            margin-bottom: 18px;
            background:
                linear-gradient(135deg,
                    rgba(79,140,255,.10),
                    rgba(255,255,255,.025) 48%,
                    rgba(36,164,124,.07));
            box-shadow: 0 18px 60px rgba(0,0,0,.06);
            animation: rise .5s ease-out both;
        }

        .hero::after {
            content: "";
            position: absolute;
            width: 320px;
            height: 320px;
            right: -115px;
            top: -155px;
            border-radius: 50%;
            border: 1px solid rgba(79,140,255,.14);
            box-shadow: 0 0 80px rgba(79,140,255,.07);
        }

        .hero-kicker {
            font-size: .74rem;
            font-weight: 800;
            letter-spacing: 1.35px;
            text-transform: uppercase;
            opacity: .63;
            margin-bottom: 11px;
        }

        .hero-title {
            margin: 0;
            font-size: clamp(2.1rem, 5vw, 3rem);
            line-height: 1.03;
            letter-spacing: -1.8px;
            font-weight: 820;
        }

        .hero-copy {
            margin: 13px 0 0 0;
            max-width: 760px;
            font-size: .98rem;
            line-height: 1.62;
            opacity: .67;
        }

        .pill-row {
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
            margin-top: 20px;
        }

        .pill {
            padding: 7px 11px;
            border: 1px solid var(--border);
            border-radius: 999px;
            font-size: .74rem;
            background: rgba(255,255,255,.035);
            opacity: .82;
        }

        .section-kicker {
            text-transform: uppercase;
            font-size: .71rem;
            font-weight: 800;
            letter-spacing: 1.3px;
            opacity: .48;
            margin-bottom: 4px;
        }

        .section-title {
            font-size: 1.42rem;
            font-weight: 760;
            letter-spacing: -.35px;
            margin-bottom: 3px;
        }

        .section-copy {
            font-size: .88rem;
            opacity: .61;
            margin-bottom: 15px;
        }

        .risk-panel {
            border: 1px solid var(--border);
            border-radius: 20px;
            padding: 22px 24px;
            background: var(--bg-card);
            min-height: 100%;
        }

        .risk-label {
            font-size: .72rem;
            text-transform: uppercase;
            letter-spacing: 1.15px;
            font-weight: 800;
            opacity: .5;
        }

        .risk-number {
            font-size: 3.4rem;
            line-height: 1;
            font-weight: 840;
            letter-spacing: -2.3px;
            margin: 8px 0 7px 0;
        }

        .risk-subtitle {
            font-size: .86rem;
            opacity: .63;
            line-height: 1.5;
        }

        .decision-good, .decision-high {
            border-radius: 13px;
            padding: 13px 15px;
            margin-top: 15px;
            font-size: .85rem;
            line-height: 1.5;
        }

        .decision-good {
            border-left: 4px solid #24a47c;
            background: rgba(36,164,124,.09);
        }

        .decision-high {
            border-left: 4px solid #d95d5d;
            background: rgba(217,93,93,.09);
        }

        .signal-box {
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 14px 15px;
            background: rgba(255,255,255,.025);
        }

        .signal-label {
            font-size: .72rem;
            opacity: .52;
            margin-bottom: 4px;
        }

        .signal-value {
            font-size: 1.28rem;
            font-weight: 760;
        }

        div[data-testid="stMetric"] {
            border: 1px solid var(--border);
            border-radius: 15px;
            padding: 15px 16px;
            background: rgba(255,255,255,.025);
        }

        div[data-testid="stMetricLabel"] {
            opacity: .62;
        }

        div[data-testid="stMetricValue"] {
            font-weight: 760;
        }

        div[data-testid="stDataEditor"] {
            border: 1px solid var(--border);
            border-radius: 15px;
            overflow: hidden;
        }

        div[data-testid="stFormSubmitButton"] > button,
        div.stButton > button {
            min-height: 49px;
            border-radius: 11px;
            font-weight: 740;
        }

        .notice {
            border: 1px solid var(--border);
            border-radius: 13px;
            padding: 12px 14px;
            background: rgba(255,255,255,.023);
            font-size: .81rem;
            opacity: .76;
            line-height: 1.55;
        }

        .footer {
            text-align: center;
            opacity: .45;
            font-size: .75rem;
            padding-top: 26px;
            line-height: 1.55;
        }

        @keyframes rise {
            from {opacity: 0; transform: translateY(10px);}
            to {opacity: 1; transform: translateY(0);}
        }

        @media (max-width: 760px) {
            .block-container {
                padding-left: 1rem;
                padding-right: 1rem;
            }

            .hero {
                padding: 25px 22px;
                border-radius: 18px;
            }

            .risk-number {
                font-size: 2.8rem;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HELPERS
# ============================================================

def section(kicker: str, title: str, copy: str):
    st.markdown(
        f"""
        <div class="section-kicker">{kicker}</div>
        <div class="section-title">{title}</div>
        <div class="section-copy">{copy}</div>
        """,
        unsafe_allow_html=True,
    )


def build_customer_data(
    limit_bal,
    sex,
    education,
    marriage,
    age,
    history_df,
):
    pay_status = (
        history_df["Repayment Status"]
        .map(STATUS_MAP)
        .astype(int)
        .tolist()
    )

    bills = history_df["Statement Balance"].astype(float).tolist()
    payments = history_df["Payment Made"].astype(float).tolist()

    customer_data = pd.DataFrame(
        [{
            "LIMIT_BAL": float(limit_bal),
            "SEX": int(sex),
            "EDUCATION": int(education),
            "MARRIAGE": int(marriage),
            "AGE": int(age),

            "PAY_0": pay_status[0],
            "PAY_2": pay_status[1],
            "PAY_3": pay_status[2],
            "PAY_4": pay_status[3],
            "PAY_5": pay_status[4],
            "PAY_6": pay_status[5],

            "BILL_AMT1": bills[0],
            "BILL_AMT2": bills[1],
            "BILL_AMT3": bills[2],
            "BILL_AMT4": bills[3],
            "BILL_AMT5": bills[4],
            "BILL_AMT6": bills[5],

            "PAY_AMT1": payments[0],
            "PAY_AMT2": payments[1],
            "PAY_AMT3": payments[2],
            "PAY_AMT4": payments[3],
            "PAY_AMT5": payments[4],
            "PAY_AMT6": payments[5],
        }]
    )

    return customer_data, pay_status, bills, payments


def validate_inputs(limit_bal, age, pay_status, bills, payments):
    errors = []

    if limit_bal <= 0:
        errors.append("Credit limit must be greater than zero.")

    if not 18 <= age <= 100:
        errors.append("Age must be between 18 and 100.")

    if any(x < -2 or x > 8 for x in pay_status):
        errors.append("A repayment status is outside the model's accepted range.")

    if any(x < 0 for x in payments):
        errors.append("Payments cannot be negative.")

    if not all(np.isfinite(x) for x in bills + payments):
        errors.append("Financial values must be finite numbers.")

    return errors


def run_assessment(customer_data):
    customer_fe = add_credit_features(customer_data)
    customer_fe = customer_fe[model_features]

    numeric_values = (
        customer_fe
        .select_dtypes(include=np.number)
        .to_numpy()
    )

    if not np.isfinite(numeric_values).all():
        raise ValueError(
            "Feature engineering produced an invalid numeric value. "
            "Please review the supplied financial data."
        )

    probability = float(model.predict_proba(customer_fe)[0, 1])
    prediction = int(probability >= threshold)

    return {
        "customer_fe": customer_fe,
        "probability": probability,
        "prediction": prediction,
        "avg_utilization": float(customer_fe["AVG_UTILIZATION"].iloc[0]),
        "max_utilization": float(customer_fe["MAX_UTILIZATION"].iloc[0]),
        "avg_delay": float(customer_fe["AVG_DELAY"].iloc[0]),
        "max_delay": int(customer_fe["MAX_DELAY"].iloc[0]),
        "delayed_months": int(customer_fe["DELAY_MONTHS"].iloc[0]),
        "avg_bill": float(customer_fe["AVG_BILL_AMT"].iloc[0]),
        "avg_payment": float(customer_fe["AVG_PAY_AMT"].iloc[0]),
        "pay_to_bill": float(customer_fe["PAY_TO_BILL_RATIO"].iloc[0]),
        "pay_to_limit": float(customer_fe["PAY_TO_LIMIT_RATIO"].iloc[0]),
    }


def risk_gauge(probability: float):
    pct = probability * 100
    threshold_pct = threshold * 100

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=pct,
            number={"suffix": "%", "font": {"size": 42}},
            title={
                "text": "Estimated default probability",
                "font": {"size": 15},
            },
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickwidth": 1,
                    "ticksuffix": "%",
                },
                "bar": {"thickness": 0.24},
                "bgcolor": "rgba(128,128,128,.05)",
                "borderwidth": 0,
                "steps": [
                    {
                        "range": [0, threshold_pct],
                        "color": "rgba(36,164,124,.09)",
                    },
                    {
                        "range": [threshold_pct, 100],
                        "color": "rgba(217,93,93,.09)",
                    },
                ],
                "threshold": {
                    "line": {"width": 4},
                    "thickness": 0.75,
                    "value": threshold_pct,
                },
            },
        )
    )

    fig.update_layout(
        height=300,
        margin=dict(l=30, r=30, t=55, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(size=13),
    )

    return fig


def monthly_financial_chart(history_df):
    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=history_df["Month"],
            y=history_df["Statement Balance"],
            name="Statement balance",
            hovertemplate="%{x}<br>Balance: %{y:,.0f}<extra></extra>",
        )
    )

    fig.add_trace(
        go.Bar(
            x=history_df["Month"],
            y=history_df["Payment Made"],
            name="Payment made",
            hovertemplate="%{x}<br>Payment: %{y:,.0f}<extra></extra>",
        )
    )

    fig.update_layout(
        barmode="group",
        height=360,
        margin=dict(l=20, r=20, t=55, b=30),
        title="Monthly balance vs payment",
        xaxis_title=None,
        yaxis_title="Amount",
        legend_title=None,
        hovermode="x unified",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )

    fig.update_yaxes(gridcolor="rgba(128,128,128,.12)")

    return fig


def repayment_status_chart(history_df):
    status_numeric = (
        history_df["Repayment Status"]
        .map(STATUS_MAP)
        .astype(int)
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=history_df["Month"],
            y=status_numeric,
            mode="lines+markers",
            name="Repayment status",
            line={"width": 3},
            marker={"size": 9},
            customdata=history_df["Repayment Status"],
            hovertemplate="%{x}<br>%{customdata}<extra></extra>",
        )
    )

    fig.add_hline(
        y=0,
        line_dash="dash",
        opacity=.45,
        annotation_text="No delay",
        annotation_position="bottom right",
    )

    fig.update_layout(
        height=360,
        margin=dict(l=20, r=20, t=55, b=30),
        title="Repayment status over six months",
        xaxis_title=None,
        yaxis_title="Delay status",
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )

    fig.update_yaxes(
        dtick=1,
        gridcolor="rgba(128,128,128,.12)",
    )

    return fig


def utilization_chart(history_df, limit_bal):
    util = np.where(
        float(limit_bal) != 0,
        history_df["Statement Balance"].astype(float) / float(limit_bal) * 100,
        0,
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=history_df["Month"],
            y=util,
            mode="lines+markers",
            fill="tozeroy",
            name="Utilization",
            line={"width": 3},
            marker={"size": 8},
            hovertemplate="%{x}<br>Utilization: %{y:.1f}%<extra></extra>",
        )
    )

    fig.add_hline(
        y=100,
        line_dash="dot",
        opacity=.45,
        annotation_text="100% of entered limit",
        annotation_position="top right",
    )

    fig.update_layout(
        height=340,
        margin=dict(l=20, r=20, t=55, b=30),
        title="Monthly utilization",
        xaxis_title=None,
        yaxis_title="Utilization (%)",
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )

    fig.update_yaxes(gridcolor="rgba(128,128,128,.12)")

    return fig


def local_shap_chart(customer_fe):
    try:
        import shap
    except ImportError:
        return None, "SHAP is not installed locally. Run: pip install shap"

    try:
        preprocessor = model.named_steps["preprocessor"]
        classifier = model.named_steps["classifier"]

        transformed = preprocessor.transform(customer_fe)
        feature_names = preprocessor.get_feature_names_out()

        explainer = shap.TreeExplainer(classifier)
        explanation = explainer(transformed)

        values = np.asarray(explanation.values)

        if values.ndim == 3:
            shap_values = values[0, :, 1]
        elif values.ndim == 2:
            shap_values = values[0]
        else:
            return None, "Unexpected SHAP output shape."

        readable_names = [
            str(name)
            .replace("num__", "")
            .replace("cat__", "")
            for name in feature_names
        ]

        shap_df = pd.DataFrame(
            {
                "Feature": readable_names,
                "Contribution": shap_values,
            }
        )

        shap_df["Absolute"] = shap_df["Contribution"].abs()
        shap_df = (
            shap_df
            .sort_values("Absolute", ascending=False)
            .head(10)
            .sort_values("Contribution")
        )

        colors = [
            "#d95d5d" if x > 0 else "#24a47c"
            for x in shap_df["Contribution"]
        ]

        fig = go.Figure(
            go.Bar(
                x=shap_df["Contribution"],
                y=shap_df["Feature"],
                orientation="h",
                marker={"color": colors},
                hovertemplate=(
                    "%{y}<br>SHAP contribution: %{x:.4f}<extra></extra>"
                ),
            )
        )

        fig.add_vline(x=0, line_width=1, opacity=.45)

        fig.update_layout(
            height=430,
            margin=dict(l=20, r=20, t=55, b=30),
            title="Top local SHAP contributions",
            xaxis_title="Contribution to default prediction",
            yaxis_title=None,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
        )

        fig.update_xaxes(gridcolor="rgba(128,128,128,.12)")

        return fig, None

    except Exception as exc:
        return None, f"SHAP explanation unavailable: {exc}"


# ============================================================
# SESSION STATE
# ============================================================

if "history_data" not in st.session_state:
    st.session_state.history_data = DEFAULT_HISTORY.copy()


# ============================================================
# HERO
# ============================================================

st.markdown(
    f"""
    <div class="hero">
        <div class="hero-kicker">Credit risk analytics platform</div>
        <h1 class="hero-title">Credit Risk Intelligence</h1>
        <p class="hero-copy">
            A production-style machine-learning interface for single-customer
            default-risk assessment, behavioural analytics, explainability,
            and model governance.
        </p>
        <div class="pill-row">
            <span class="pill">Model v{model_version}</span>
            <span class="pill">Threshold {threshold * 100:.0f}%</span>
            <span class="pill">Random Forest</span>
            <span class="pill">SHAP-ready</span>
            <span class="pill">Human oversight required</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# TOP CONTROLS
# ============================================================

control_1, control_2, control_3 = st.columns([1, 1, 4])

with control_1:
    if st.button("Load demo account", use_container_width=True):
        st.session_state.history_data = DEMO_HISTORY.copy()
        st.rerun()

with control_2:
    if st.button("Clear history", use_container_width=True):
        st.session_state.history_data = DEFAULT_HISTORY.copy()
        st.session_state.pop("assessment", None)
        st.rerun()

with control_3:
    st.caption(
        "Tip: balances and payments start at zero so every assessment reflects "
        "the customer data you enter. Use the demo account only when needed."
    )


# ============================================================
# INPUT
# ============================================================

with st.form("risk_assessment_form"):
    section(
        "01 · Customer profile",
        "Customer information",
        "Enter the profile information used by the trained credit-risk model.",
    )

    p1, p2, p3, p4, p5 = st.columns([1.25, .75, .9, 1.2, 1.05])

    with p1:
        limit_bal = st.number_input(
            "Credit Limit",
            min_value=10000.0,
            max_value=1000000.0,
            value=200000.0,
            step=10000.0,
            help="Entered account credit limit.",
        )

    with p2:
        age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=30,
            step=1,
        )

    with p3:
        sex = st.selectbox(
            "Sex",
            [1, 2],
            format_func=lambda x: "Male" if x == 1 else "Female",
        )

    with p4:
        education = st.selectbox(
            "Education",
            [1, 2, 3, 4],
            format_func=lambda x: {
                1: "Graduate School",
                2: "University",
                3: "High School",
                4: "Other",
            }[x],
        )

    with p5:
        marriage = st.selectbox(
            "Marital Status",
            [1, 2, 3],
            format_func=lambda x: {
                1: "Married",
                2: "Single",
                3: "Other",
            }[x],
        )

    st.write("")
    st.divider()

    section(
        "02 · Credit behaviour",
        "Six-month account history",
        "Record repayment status, monthly statement balance, and payment made.",
    )

    st.markdown(
        """
        <div class="notice">
            <b>Repayment status:</b> positive values represent months of delay.
            “No Delay” is coded as 0. Special statuses preserve the original
            UCI dataset coding used during model training.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")

    edited_history = st.data_editor(
        st.session_state.history_data,
        hide_index=True,
        use_container_width=True,
        disabled=["Month"],
        num_rows="fixed",
        key="credit_history_editor",
        column_config={
            "Month": st.column_config.TextColumn(
                "Month",
                width="small",
            ),
            "Repayment Status": st.column_config.SelectboxColumn(
                "Repayment Status",
                options=list(STATUS_MAP.keys()),
                required=True,
                width="medium",
            ),
            "Statement Balance": st.column_config.NumberColumn(
                "Statement Balance",
                step=1000.0,
                format="%.0f",
                width="medium",
            ),
            "Payment Made": st.column_config.NumberColumn(
                "Payment Made",
                min_value=0.0,
                step=1000.0,
                format="%.0f",
                width="medium",
            ),
        },
    )

    st.write("")

    submitted = st.form_submit_button(
        "Run Credit Risk Assessment",
        type="primary",
        use_container_width=True,
    )


# ============================================================
# PREDICTION
# ============================================================

if submitted:
    st.session_state.history_data = edited_history.copy()

    customer_data, pay_status, bills, payments = build_customer_data(
        limit_bal=limit_bal,
        sex=sex,
        education=education,
        marriage=marriage,
        age=age,
        history_df=edited_history,
    )

    errors = validate_inputs(
        limit_bal,
        age,
        pay_status,
        bills,
        payments,
    )

    if errors:
        st.error("Please review the supplied customer information.")
        for error in errors:
            st.write(f"- {error}")
        st.stop()

    try:
        assessment = run_assessment(customer_data)
    except Exception as exc:
        st.error(f"Assessment could not be completed: {exc}")
        st.stop()

    st.session_state.assessment = assessment
    st.session_state.customer_data = customer_data
    st.session_state.assessment_history = edited_history.copy()
    st.session_state.assessment_limit = float(limit_bal)

    st.toast("Assessment completed.")


# ============================================================
# RESULTS
# ============================================================

if "assessment" in st.session_state:
    result = st.session_state.assessment
    history_for_result = st.session_state.assessment_history
    limit_for_result = st.session_state.assessment_limit

    probability = result["probability"]
    prediction = result["prediction"]
    distance_pp = abs(probability - threshold) * 100

    st.write("")
    st.divider()

    section(
        "03 · Risk assessment",
        "Decision-support overview",
        "The model output below is a risk estimate, not an automatic lending decision.",
    )

    result_left, result_right = st.columns([.9, 1.3], gap="large")

    with result_left:
        st.markdown(
            f"""
            <div class="risk-panel">
                <div class="risk-label">Default probability</div>
                <div class="risk-number">{probability * 100:.1f}%</div>
                <div class="risk-subtitle">
                    Model threshold: {threshold * 100:.0f}% ·
                    Distance from threshold: {distance_pp:.1f} percentage points
                </div>
                <div class="{'decision-high' if prediction else 'decision-good'}">
                    <b>{"Higher predicted default risk" if prediction else "Lower predicted default risk"}</b><br>
                    {"The estimate is above the model threshold and should receive additional human review."
                     if prediction else
                     "The estimate is below the model threshold. This does not constitute loan approval."}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.plotly_chart(
            risk_gauge(probability),
            use_container_width=True,
            config={"displayModeBar": False},
        )

    with result_right:
        k1, k2 = st.columns(2)
        k1.metric(
            "Average utilization",
            f"{result['avg_utilization'] * 100:.1f}%",
        )
        k2.metric(
            "Delayed months",
            result["delayed_months"],
        )

        k3, k4 = st.columns(2)
        k3.metric(
            "Maximum delay",
            f"{result['max_delay']} mo."
            if result["max_delay"] > 0
            else "None",
        )
        k4.metric(
            "Average payment",
            f"{result['avg_payment']:,.0f}",
        )

        k5, k6 = st.columns(2)
        k5.metric(
            "Average bill",
            f"{result['avg_bill']:,.0f}",
        )
        k6.metric(
            "Payment / bill",
            f"{result['pay_to_bill']:.3f}",
        )

        if result["avg_utilization"] > 1:
            st.warning(
                f"Average utilization is "
                f"{result['avg_utilization'] * 100:.1f}%. "
                "The average statement balance is higher than the entered "
                "credit limit. This can occur in the source data and is not "
                "capped because the model was trained on the original ratios."
            )
        else:
            st.caption(
                f"Average statement balance uses "
                f"{result['avg_utilization'] * 100:.1f}% of the entered "
                "credit limit."
            )

    st.write("")
    section(
        "04 · Behaviour analytics",
        "Customer credit behaviour",
        "Interactive charts summarize the six-month account history used in this assessment.",
    )

    chart_1, chart_2 = st.columns(2, gap="large")

    with chart_1:
        st.plotly_chart(
            monthly_financial_chart(history_for_result),
            use_container_width=True,
            config={"displayModeBar": False},
        )

    with chart_2:
        st.plotly_chart(
            repayment_status_chart(history_for_result),
            use_container_width=True,
            config={"displayModeBar": False},
        )

    st.plotly_chart(
        utilization_chart(history_for_result, limit_for_result),
        use_container_width=True,
        config={"displayModeBar": False},
    )

    st.write("")
    section(
        "05 · Explainability",
        "Why did the model move toward this prediction?",
        "SHAP values estimate each processed feature's contribution to the current default prediction.",
    )

    shap_fig, shap_error = local_shap_chart(result["customer_fe"])

    if shap_fig is not None:
        st.plotly_chart(
            shap_fig,
            use_container_width=True,
            config={"displayModeBar": False},
        )
        st.caption(
            "Positive SHAP contributions push the model toward default; "
            "negative contributions push it away from default. "
            "These are model explanations, not causal effects."
        )
    else:
        st.info(shap_error)

    with st.expander("View engineered account indicators"):
        indicator_df = pd.DataFrame(
            {
                "Indicator": [
                    "Average utilization",
                    "Maximum utilization",
                    "Average delay",
                    "Maximum delay",
                    "Delayed months",
                    "Average bill amount",
                    "Average payment amount",
                    "Payment-to-bill ratio",
                    "Payment-to-limit ratio",
                ],
                "Value": [
                    f"{result['avg_utilization'] * 100:.2f}%",
                    f"{result['max_utilization'] * 100:.2f}%",
                    f"{result['avg_delay']:.2f}",
                    f"{result['max_delay']}",
                    f"{result['delayed_months']}",
                    f"{result['avg_bill']:,.0f}",
                    f"{result['avg_payment']:,.0f}",
                    f"{result['pay_to_bill']:.3f}",
                    f"{result['pay_to_limit']:.3f}",
                ],
            }
        )

        st.dataframe(
            indicator_df,
            hide_index=True,
            use_container_width=True,
        )


# ============================================================
# MODEL CARD
# ============================================================

st.write("")
st.divider()

section(
    "Model governance",
    "Model card & validation snapshot",
    "Core model information and held-out test metrics from the saved governance report.",
)

mc1, mc2, mc3, mc4 = st.columns(4)

mc1.metric(
    "Model",
    "Random Forest",
)

mc2.metric(
    "Version",
    str(model_version),
)

mc3.metric(
    "Threshold",
    f"{threshold * 100:.0f}%",
)

mc4.metric(
    "Explainability",
    "SHAP",
)

if report:
    perf1, perf2, perf3, perf4 = st.columns(4)

    perf1.metric(
        "ROC-AUC",
        f"{float(report.get('Test_ROC_AUC', 0)):.3f}",
    )
    perf2.metric(
        "PR-AUC",
        f"{float(report.get('Test_PR_AUC', 0)):.3f}",
    )
    perf3.metric(
        "Recall",
        f"{float(report.get('Test_Recall', 0)):.3f}",
    )
    perf4.metric(
        "F1",
        f"{float(report.get('Test_F1', 0)):.3f}",
    )

    with st.expander("Governance details"):
        gov = report.get("Governance", {})

        gov_df = pd.DataFrame(
            {
                "Item": [
                    "Dataset",
                    "Dataset size",
                    "Processed features",
                    "Human oversight required",
                    "Automatic loan approval",
                    "Fairness audit performed",
                    "Threshold tuned on test set",
                    "Important limitation",
                ],
                "Value": [
                    report.get("Dataset", "N/A"),
                    report.get("Dataset_Size", "N/A"),
                    report.get("Processed_Features", "N/A"),
                    gov.get("Human_Oversight_Required", "N/A"),
                    gov.get("Automatic_Loan_Approval", "N/A"),
                    gov.get("Fairness_Audit_Performed", "N/A"),
                    gov.get("Threshold_Tuned_On_Test_Set", "N/A"),
                    report.get("Important_Limitation", "N/A"),
                ],
            }
        )

        st.dataframe(
            gov_df,
            hide_index=True,
            use_container_width=True,
        )
else:
    st.info(
        "Governance report not found at "
        "`reports/credit_risk_regulatory_report.json`."
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        Credit Risk Intelligence · ML decision-support interface
        <br>
        Educational and analytical use only · Human review required for lending decisions
    </div>
    """,
    unsafe_allow_html=True,
)
'''

path = Path("/mnt/data/app_world_standard.py")
path.write_text(code, encoding="utf-8")

# Syntax check
compile(code, str(path), "exec")

print(f"Created: {path}")
print(f"Lines: {len(code.splitlines())}")
print("Syntax check: PASSED")
