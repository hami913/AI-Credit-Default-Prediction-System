"""Credit Risk Intelligence - Streamlit front end for the credit default model."""

from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.feature_engineering import add_credit_features

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Credit Risk Intelligence",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "credit_default_model_v1.joblib"


# ============================================================
# LOAD ASSETS
# ============================================================
@st.cache_resource
def load_model_artifact() -> Dict[str, Any]:
    """Load the saved model artifact from disk (cached across reruns)."""
    return joblib.load(MODEL_PATH)


try:
    artifact = load_model_artifact()
except FileNotFoundError:
    st.error(f"Model file not found: `{MODEL_PATH}`")
    st.stop()
except Exception as exc:
    st.error(f"Could not load model: {exc}")
    st.stop()

model = artifact["pipeline"]
threshold = float(artifact.get("threshold", 0.50))
model_features = artifact["features"]

# ============================================================
# CONSTANTS
# ============================================================
MONTHS = [
    "Most recent",
    "1 month ago",
    "2 months ago",
    "3 months ago",
    "4 months ago",
    "5 months ago",
]

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

# Chart palette (kept in one place so charts match the theme)
INDIGO = "#7C3AED"
AQUA = "#19D3C5"
CORAL = "#FF5C7A"
MINT = "#22C58B"
INK = "#0E1330"

# ============================================================
# STYLE
# ============================================================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Sora:wght@600;700;800&display=swap');

    :root {
        --ink: #0E1330;
        --ink-2: #1A2150;
        --indigo: #7C3AED;
        --violet: #A855F7;
        --aqua: #19D3C5;
        --coral: #FF5C7A;
        --mint: #22C58B;
        --amber: #FFB020;
        --text: #1B2140;
        --muted: #6B7391;
        --line: #E6E8F4;
        --surface: #FFFFFF;
        --bg: #F3EEFF;
        --shadow-lg: 0 30px 70px -20px rgba(76, 29, 149, .28);
        --shadow-md: 0 14px 34px -14px rgba(76, 29, 149, .20);
        --shadow-sm: 0 6px 18px -10px rgba(76, 29, 149, .18);
        --font-head: 'Sora', 'Inter', sans-serif;
        --font-body: 'Inter', system-ui, sans-serif;
    }

    html {scroll-behavior: smooth;}

    html, body, .stApp, [class*="css"] {
        font-family: var(--font-body);
    }

    .stApp {
        background:
            radial-gradient(60rem 28rem at 0% -5%, rgba(124,58,237,.13), transparent 60%),
            radial-gradient(50rem 26rem at 100% 0%, rgba(25,211,197,.13), transparent 60%),
            var(--bg);
        color: var(--text);
    }

    .block-container {
        max-width: 1180px;
        padding-top: 1.2rem;
        padding-bottom: 3rem;
    }

    #MainMenu, footer, header[data-testid="stHeader"] {visibility: hidden;}

    /* ---------- Hero ---------- */
    .hero {
        position: relative;
        overflow: hidden;
        border-radius: 30px;
        padding: 46px 48px 40px;
        color: #FFFFFF;
        background:
            radial-gradient(34rem 20rem at 88% 8%, rgba(168,85,247,.60), transparent 65%),
            radial-gradient(28rem 18rem at 70% 110%, rgba(25,211,197,.42), transparent 65%),
            radial-gradient(24rem 16rem at 0% 100%, rgba(124,58,237,.55), transparent 65%),
            linear-gradient(135deg, #160B36 0%, #2A1257 60%, #3B1B7A 100%);
        box-shadow: var(--shadow-lg);
        margin-bottom: 1.4rem;
        isolation: isolate;
    }

    .hero::before {
        content: "";
        position: absolute;
        inset: 0;
        z-index: -1;
        background-image:
            linear-gradient(rgba(255,255,255,.055) 1px, transparent 1px),
            linear-gradient(90deg, rgba(255,255,255,.055) 1px, transparent 1px);
        background-size: 44px 44px;
        mask-image: radial-gradient(ellipse at 75% 30%, #000 0%, transparent 70%);
        -webkit-mask-image: radial-gradient(ellipse at 75% 30%, #000 0%, transparent 70%);
    }

    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: .55rem;
        padding: .42rem .85rem;
        border-radius: 999px;
        background: rgba(255,255,255,.10);
        border: 1px solid rgba(255,255,255,.18);
        backdrop-filter: blur(10px);
        color: #E6EBFF;
        font-size: .8rem;
        font-weight: 600;
        margin-bottom: 1.1rem;
    }

    .hero-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: var(--aqua);
        box-shadow: 0 0 0 0 rgba(25,211,197,.7);
        animation: pulse 2.2s infinite;
    }

    @keyframes pulse {
        0% {box-shadow: 0 0 0 0 rgba(25,211,197,.65);}
        70% {box-shadow: 0 0 0 10px rgba(25,211,197,0);}
        100% {box-shadow: 0 0 0 0 rgba(25,211,197,0);}
    }

    .hero-title {
        font-family: var(--font-head);
        color: #FFFFFF;
        font-size: clamp(2.3rem, 5vw, 3.7rem);
        font-weight: 800;
        letter-spacing: -.045em;
        line-height: 1.04;
        margin: 0;
        padding: 0;
        max-width: 760px;
        background: linear-gradient(100deg, #FFFFFF 30%, #E2CCFF 70%, #8CF3E8 100%);
        -webkit-background-clip: text;
        background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero-copy {
        color: #C4CCEE;
        font-size: 1.05rem;
        margin: 1rem 0 0;
        max-width: 620px;
        line-height: 1.65;
    }

    .hero-meta {
        display: flex;
        flex-wrap: wrap;
        gap: .6rem;
        margin-top: 1.5rem;
    }

    .hero-meta span {
        display: inline-block;
        padding: .45rem .8rem;
        border-radius: 12px;
        background: rgba(255,255,255,.08);
        color: #E3E8FF;
        font-size: .82rem;
        font-weight: 500;
        border: 1px solid rgba(255,255,255,.12);
        backdrop-filter: blur(8px);
    }

    /* ---------- Section headings ---------- */
    .section-wrap {
        display: flex;
        align-items: flex-start;
        gap: .95rem;
        margin: .2rem 0 1rem;
    }

    .section-step {
        flex: 0 0 auto;
        width: 38px;
        height: 38px;
        display: grid;
        place-items: center;
        border-radius: 12px;
        color: #FFFFFF;
        font-family: var(--font-head);
        font-weight: 700;
        font-size: .95rem;
        background: linear-gradient(135deg, var(--indigo), var(--violet));
        box-shadow: 0 10px 20px -8px rgba(124,58,237,.65);
    }

    .section-title {
        font-family: var(--font-head);
        color: var(--text);
        font-size: 1.32rem;
        font-weight: 700;
        letter-spacing: -.02em;
        line-height: 1.2;
    }

    .section-copy {
        color: var(--muted);
        font-size: .9rem;
        margin-top: .2rem;
        line-height: 1.55;
    }

    /* ---------- Form / cards ---------- */
    div[data-testid="stForm"] {
        background: rgba(255,255,255,.86);
        border: 1px solid rgba(255,255,255,.9);
        outline: 1px solid var(--line);
        border-radius: 26px;
        padding: 1.6rem 1.7rem 1.5rem;
        box-shadow: var(--shadow-md);
        backdrop-filter: blur(14px);
    }

    div[data-testid="stExpander"] {
        background: rgba(255,255,255,.8);
        border: 1px solid var(--line);
        border-radius: 16px;
        overflow: hidden;
        box-shadow: var(--shadow-sm);
        margin-bottom: 1.2rem;
    }

    div[data-testid="stExpander"] details summary p {
        font-weight: 600;
        color: var(--text);
    }

    div[data-testid="stNumberInput"] input,
    div[data-testid="stSelectbox"] > div > div {
        border-radius: 12px !important;
        border-color: var(--line) !important;
        background: #FBFBFF !important;
    }

    /* Readable labels and input text on the light form card (any Streamlit theme) */
    div[data-testid="stWidgetLabel"] p,
    div[data-testid="stWidgetLabel"] label,
    div[data-testid="stForm"] label p {
        color: var(--text) !important;
        font-weight: 600 !important;
    }

    div[data-testid="stNumberInput"] div[data-baseweb="input"],
    div[data-testid="stNumberInput"] div[data-baseweb="base-input"] {
        background: #FBFBFF !important;
        border-radius: 12px !important;
        border-color: var(--line) !important;
    }

    div[data-testid="stNumberInput"] input,
    div[data-testid="stSelectbox"] div[data-baseweb="select"] *,
    div[data-testid="stSelectbox"] input {
        color: var(--text) !important;
        -webkit-text-fill-color: var(--text) !important;
    }

    div[data-testid="stNumberInput"] button {
        background: #EEF0FA !important;
        color: var(--text) !important;
        border: 0 !important;
    }

    div[data-testid="stSelectbox"] svg {
        fill: var(--muted) !important;
    }

    div[data-testid="stNumberInput"] input:focus,
    div[data-testid="stSelectbox"] > div > div:focus-within {
        border-color: var(--indigo) !important;
        box-shadow: 0 0 0 3px rgba(124,58,237,.16) !important;
    }

    div[data-testid="stDataEditor"] {
        border: 1px solid var(--line);
        border-radius: 16px;
        overflow: hidden;
    }

    div[data-testid="stFormSubmitButton"] button[kind="primary"] {
        background: linear-gradient(95deg, var(--indigo) 0%, var(--violet) 55%, #C084FC 100%);
        color: #FFFFFF;
        border: 0;
        border-radius: 14px;
        min-height: 54px;
        font-family: var(--font-head);
        font-weight: 700;
        font-size: 1rem;
        letter-spacing: .005em;
        box-shadow: 0 18px 34px -14px rgba(124,58,237,.75);
        transition: transform .18s ease, box-shadow .18s ease, filter .18s ease;
    }

    div[data-testid="stFormSubmitButton"] button[kind="primary"]:hover {
        transform: translateY(-2px);
        filter: brightness(1.06);
        box-shadow: 0 22px 38px -14px rgba(124,58,237,.85);
    }

    div[data-testid="stFormSubmitButton"] button[kind="primary"]:focus-visible,
    .stButton > button:focus-visible {
        outline: 3px solid rgba(124,58,237,.4);
        outline-offset: 2px;
    }

    .stButton > button {
        border-radius: 12px;
        border: 1px solid var(--line);
        background: #FFFFFF;
        font-weight: 600;
        color: var(--text);
        transition: border-color .15s ease, color .15s ease;
    }

    .stButton > button:hover {
        border-color: var(--indigo);
        color: var(--indigo);
    }

    /* ---------- Result ---------- */
    .result-shell {
        position: relative;
        overflow: hidden;
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 26px;
        padding: 28px 30px;
        box-shadow: var(--shadow-lg);
        min-height: 320px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }

    .result-shell::before {
        content: "";
        position: absolute;
        width: 320px;
        height: 320px;
        right: -120px;
        top: -140px;
        border-radius: 50%;
        opacity: .18;
        filter: blur(6px);
    }

    .result-shell.low::before {background: radial-gradient(circle, var(--mint), transparent 70%);}
    .result-shell.high::before {background: radial-gradient(circle, var(--coral), transparent 70%);}

    .result-top {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: .8rem;
        flex-wrap: wrap;
    }

    .result-eyebrow {
        color: var(--muted);
        font-size: .88rem;
        font-weight: 600;
    }

    .result-status {
        display: inline-flex;
        align-items: center;
        gap: .45rem;
        border-radius: 999px;
        padding: .4rem .8rem;
        font-size: .82rem;
        font-weight: 700;
    }

    .result-shell.low .result-status {
        color: #0A7A54;
        background: #E6F8F0;
        border: 1px solid #C4EEDB;
    }

    .result-shell.high .result-status {
        color: #C2264A;
        background: #FFEDF1;
        border: 1px solid #FFCFDA;
    }

    .result-number {
        font-family: var(--font-head);
        font-size: clamp(3.6rem, 7vw, 5.6rem);
        font-weight: 800;
        letter-spacing: -.06em;
        line-height: 1;
        margin: .7rem 0 1.1rem;
    }

    .result-shell.low .result-number {
        background: linear-gradient(120deg, #0E9F6E, #19D3C5);
        -webkit-background-clip: text;
        background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .result-shell.high .result-number {
        background: linear-gradient(120deg, #FF5C7A, #FF8A5C);
        -webkit-background-clip: text;
        background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .meter {
        position: relative;
        height: 12px;
        border-radius: 999px;
        background: #EEF0FA;
        overflow: visible;
    }

    .meter-fill {
        height: 100%;
        border-radius: 999px;
    }

    .result-shell.low .meter-fill {background: linear-gradient(90deg, #22C58B, #19D3C5);}
    .result-shell.high .meter-fill {background: linear-gradient(90deg, #FF8A5C, #FF5C7A);}

    .meter-mark {
        position: absolute;
        top: -5px;
        width: 3px;
        height: 22px;
        border-radius: 3px;
        background: var(--ink);
        transform: translateX(-50%);
    }

    .meter-scale {
        display: flex;
        justify-content: space-between;
        color: var(--muted);
        font-size: .76rem;
        font-weight: 500;
        margin-top: .55rem;
    }

    .result-copy {
        color: var(--muted);
        font-size: .88rem;
        line-height: 1.6;
        margin-top: 1rem;
        max-width: 540px;
    }

    /* ---------- Metrics ---------- */
    div[data-testid="stMetric"] {
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 18px;
        padding: 18px 20px;
        box-shadow: var(--shadow-sm);
        position: relative;
        overflow: hidden;
    }

    div[data-testid="stMetric"]::before {
        content: "";
        position: absolute;
        left: 0;
        top: 16px;
        bottom: 16px;
        width: 4px;
        border-radius: 0 4px 4px 0;
        background: linear-gradient(180deg, var(--indigo), var(--aqua));
    }

    div[data-testid="stMetricLabel"] {
        color: var(--muted);
        font-weight: 600;
    }

    div[data-testid="stMetricValue"] {
        font-family: var(--font-head);
        color: var(--text);
        font-weight: 700;
        letter-spacing: -.03em;
    }

    /* ---------- Tabs / charts ---------- */
    div[data-baseweb="tab-list"] {
        gap: .4rem;
        background: #E9EBF8;
        padding: .3rem;
        border-radius: 14px;
        width: fit-content;
    }

    button[data-baseweb="tab"] {
        font-weight: 600;
        border-radius: 10px;
        padding: .45rem 1.1rem;
        height: auto;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        background: #FFFFFF;
        color: var(--indigo);
        box-shadow: var(--shadow-sm);
    }

    div[data-baseweb="tab-highlight"], div[data-baseweb="tab-border"] {
        display: none;
    }

    div[data-testid="stPlotlyChart"] {
        background: #FFFFFF;
        border: 1px solid var(--line);
        border-radius: 20px;
        padding: .4rem;
        box-shadow: var(--shadow-sm);
    }

    .helper-note {
        background: linear-gradient(90deg, rgba(124,58,237,.08), rgba(25,211,197,.06));
        border: 1px solid rgba(124,58,237,.18);
        color: #4A5275;
        border-radius: 14px;
        padding: .8rem 1rem;
        font-size: .85rem;
        line-height: 1.55;
        margin: .2rem 0 1rem;
    }

    .footer-note {
        color: #8A91AD;
        font-size: .8rem;
        text-align: center;
        padding: 2rem 0 .2rem;
    }

    hr {
        border-color: var(--line) !important;
        margin: 1.4rem 0 !important;
    }

    @media (max-width: 700px) {
        .block-container {
            padding-left: .9rem;
            padding-right: .9rem;
            padding-top: .8rem;
        }

        .hero {
            padding: 30px 22px 28px;
            border-radius: 22px;
        }

        div[data-testid="stForm"] {
            padding: 1rem .9rem 1.1rem;
            border-radius: 20px;
        }

        .result-shell {
            min-height: unset;
            padding: 22px;
            border-radius: 20px;
        }
    }

    @media (prefers-reduced-motion: reduce) {
        .hero-dot {animation: none;}
        * {transition: none !important;}
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HELPERS
# ============================================================
def section(step: str, title: str, copy: str) -> None:
    """Render a numbered section heading."""
    st.markdown(
        f"""
        <div class="section-wrap">
            <div class="section-step">{step}</div>
            <div>
                <div class="section-title">{title}</div>
                <div class="section-copy">{copy}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_customer_data(
    limit_bal: float,
    sex: int,
    education: int,
    marriage: int,
    age: int,
    history_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, List[int], List[float], List[float]]:
    """Convert form inputs into the single-row frame the model expects."""
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


def validate_inputs(
    limit_bal: float,
    age: int,
    pay_status: List[int],
    bills: List[float],
    payments: List[float],
) -> List[str]:
    """Return a list of validation error messages (empty when valid)."""
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


def run_assessment(customer_data: pd.DataFrame) -> Dict[str, Any]:
    """Engineer features, score the customer, and collect summary metrics."""
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


def polish_chart(
    fig: go.Figure,
    title: str,
    y_title: str = None,
    height: int = 350,
) -> go.Figure:
    """Apply the shared chart theme."""
    fig.update_layout(
        height=height,
        margin=dict(l=20, r=20, t=58, b=25),
        title=dict(
            text=title,
            x=0.02,
            xanchor="left",
            font=dict(size=17, color=INK, family="Sora, Inter, sans-serif"),
        ),
        font=dict(family="Inter, Arial, sans-serif", color="#5B6485", size=12),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hoverlabel=dict(bgcolor=INK, font_color="#FFFFFF", bordercolor=INK),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(0,0,0,0)",
        ),
    )
    fig.update_xaxes(
        showgrid=False,
        zeroline=False,
        tickfont=dict(color="#7A82A1"),
        linecolor="#E6E8F4",
    )
    fig.update_yaxes(
        title=y_title,
        gridcolor="rgba(120,130,170,.14)",
        zeroline=False,
        tickfont=dict(color="#7A82A1"),
    )
    return fig


def monthly_financial_chart(history_df: pd.DataFrame) -> go.Figure:
    """Grouped bars: statement balance versus payment per month."""
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=history_df["Month"],
            y=history_df["Statement Balance"],
            name="Statement balance",
            marker=dict(color=INDIGO, cornerradius=8),
            hovertemplate="%{x}<br>Balance: %{y:,.0f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            x=history_df["Month"],
            y=history_df["Payment Made"],
            name="Payment made",
            marker=dict(color=AQUA, cornerradius=8),
            hovertemplate="%{x}<br>Payment: %{y:,.0f}<extra></extra>",
        )
    )
    fig.update_layout(
        barmode="group",
        bargap=.28,
        bargroupgap=.08,
        xaxis_title=None,
        yaxis_title=None,
        hovermode="x unified",
    )
    polish_chart(fig, "Balance vs payment", "Amount", 355)
    return fig


def repayment_status_chart(history_df: pd.DataFrame) -> go.Figure:
    """Line chart of the repayment status code per month."""
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
            line={"width": 3, "color": "#A855F7", "shape": "spline"},
            marker={"size": 10, "color": "#FFFFFF", "line": {"width": 3, "color": "#A855F7"}},
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
        xaxis_title=None,
        yaxis_title=None,
        showlegend=False,
    )
    polish_chart(fig, "Repayment status", "Delay status", 355)
    fig.update_yaxes(dtick=1)
    return fig


def utilization_chart(history_df: pd.DataFrame, limit_bal: float) -> go.Figure:
    """Area chart of statement balance as a percentage of the credit limit."""
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
            fillcolor="rgba(124,58,237,.14)",
            name="Utilization",
            line={"width": 3, "color": INDIGO, "shape": "spline"},
            marker={"size": 9, "color": "#FFFFFF", "line": {"width": 3, "color": INDIGO}},
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
        xaxis_title=None,
        yaxis_title=None,
        showlegend=False,
    )
    polish_chart(fig, "Credit utilization", "Utilization (%)", 340)
    return fig


def risk_gauge_chart(probability: float, threshold: float) -> go.Figure:
    """Semi-circular gauge showing the probability against the threshold."""
    pct = probability * 100
    threshold_pct = threshold * 100
    bar_color = CORAL if probability >= threshold else MINT

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=pct,
            number={
                "suffix": "%",
                "font": {"size": 44, "color": INK, "family": "Sora, Inter, sans-serif"},
                "valueformat": ".1f",
            },
            title={
                "text": "Default probability",
                "font": {"size": 14, "color": "#6B7391"},
            },
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickwidth": 0,
                    "tickcolor": "#CBD5E1",
                    "tickfont": {"size": 10, "color": "#8A91AD"},
                },
                "bar": {"color": bar_color, "thickness": .32},
                "bgcolor": "#EEF0FA",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, threshold_pct], "color": "#E6F8F0"},
                    {"range": [threshold_pct, 100], "color": "#FFEDF1"},
                ],
                "threshold": {
                    "line": {"color": INK, "width": 3},
                    "thickness": .8,
                    "value": threshold_pct,
                },
            },
        )
    )
    fig.update_layout(
        height=320,
        margin=dict(l=32, r=32, t=56, b=15),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, Arial, sans-serif"),
    )
    return fig


def local_shap_chart(customer_fe: pd.DataFrame):
    """Build a SHAP bar chart for one customer; returns (figure, error)."""
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
            str(name).replace("num__", "").replace("cat__", "")
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

        colors = [CORAL if x > 0 else MINT for x in shap_df["Contribution"]]

        fig = go.Figure(
            go.Bar(
                x=shap_df["Contribution"],
                y=shap_df["Feature"],
                orientation="h",
                marker={"color": colors, "cornerradius": 6},
                hovertemplate="%{y}<br>SHAP contribution: %{x:.4f}<extra></extra>",
            )
        )
        fig.add_vline(x=0, line_width=1, opacity=.45)
        fig.update_layout(
            height=430,
            margin=dict(l=20, r=20, t=55, b=30),
            title=dict(
                text="Top factors influencing this prediction",
                x=.02,
                xanchor="left",
                font=dict(size=17, color=INK, family="Sora, Inter, sans-serif"),
            ),
            xaxis_title="SHAP contribution",
            yaxis_title=None,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
        )
        fig.update_xaxes(gridcolor="rgba(120,130,170,.14)")
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
    """
    <div class="hero">
        <div class="hero-badge"><span class="hero-dot"></span>AI credit risk assessment</div>
        <h1 class="hero-title">Know the risk before the next statement.</h1>
        <p class="hero-copy">
            Enter a customer profile and six months of repayment behavior to estimate
            next-month default probability, with a clear explanation of what drives it.
        </p>
        <div class="hero-meta">
            <span>📈 Next-month risk</span>
            <span>🗓️ 6-month payment history</span>
            <span>🔍 Explainable prediction</span>
            <span>🧑‍⚖️ Human-reviewed</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.expander("Demo controls"):
    demo_col, reset_col = st.columns(2)
    with demo_col:
        if st.button("Load sample history", use_container_width=True):
            st.session_state.history_data = DEMO_HISTORY.copy()
            st.session_state.pop("assessment", None)
            st.session_state.pop("credit_history_editor", None)
            st.rerun()
    with reset_col:
        if st.button("Clear history", use_container_width=True):
            st.session_state.history_data = DEFAULT_HISTORY.copy()
            st.session_state.pop("assessment", None)
            st.session_state.pop("credit_history_editor", None)
            st.rerun()

    st.caption(
        "Sample values are for interface testing only. Use real account values for an actual assessment."
    )

# ============================================================
# INPUT
# ============================================================
with st.form("risk_assessment_form"):
    section(
        "1",
        "Customer information",
        "Basic demographic and credit-limit inputs used by the model.",
    )

    p1, p2, p3 = st.columns(3)
    with p1:
        limit_bal = st.number_input(
            "Credit limit",
            min_value=10000.0,
            max_value=1000000.0,
            value=200000.0,
            step=10000.0,
        )
    with p2:
        age = st.number_input("Age", min_value=18, max_value=100, value=30, step=1)
    with p3:
        sex = st.selectbox(
            "Sex",
            [1, 2],
            format_func=lambda x: "Male" if x == 1 else "Female",
        )

    p4, p5 = st.columns(2)
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
            "Marital status",
            [1, 2, 3],
            format_func=lambda x: {1: "Married", 2: "Single", 3: "Other"}[x],
        )

    st.write("")
    st.divider()

    section(
        "2",
        "Six-month account history",
        "Add the six most recent repayment statuses, statement balances, and payments.",
    )

    st.markdown(
        '<div class="helper-note"><b>Tip:</b> Positive repayment-status values '
        "represent months of delay. Special statuses keep the dataset's original codes.</div>",
        unsafe_allow_html=True,
    )

    edited_history = st.data_editor(
        st.session_state.history_data,
        hide_index=True,
        use_container_width=True,
        disabled=["Month"],
        num_rows="fixed",
        key="credit_history_editor",
        column_config={
            "Month": st.column_config.TextColumn("Month", width="small"),
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
        "Run credit risk assessment",
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

    errors = validate_inputs(limit_bal, age, pay_status, bills, payments)

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

    st.write("")
    section(
        "3",
        "Estimated default risk",
        "The model output comes first; supporting behavior and explanation are below.",
    )

    fill_pct = min(max(probability * 100, 0.0), 100.0)
    mark_pct = min(max(threshold * 100, 0.0), 100.0)

    summary_col, gauge_col = st.columns([1.25, .9], gap="large")

    with summary_col:
        st.markdown(
            f"""
            <div class="result-shell {'high' if prediction else 'low'}">
                <div class="result-top">
                    <div class="result-eyebrow">Next-month default probability</div>
                    <div class="result-status">
                        {'● Higher predicted risk' if prediction else '● Lower predicted risk'}
                    </div>
                </div>
                <div class="result-number">{probability * 100:.1f}%</div>
                <div class="meter">
                    <div class="meter-fill" style="width:{fill_pct:.1f}%"></div>
                    <div class="meter-mark" style="left:{mark_pct:.1f}%"></div>
                </div>
                <div class="meter-scale">
                    <span>0%</span>
                    <span>Threshold {threshold * 100:.0f}%</span>
                    <span>100%</span>
                </div>
                <div class="result-copy">
                    Classification is based on the saved model threshold of <b>{threshold * 100:.0f}%</b>.
                    This prediction is decision support only and should be reviewed by a qualified person.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with gauge_col:
        st.plotly_chart(
            risk_gauge_chart(probability, threshold),
            use_container_width=True,
            config={"displayModeBar": False},
        )

    st.write("")
    m1, m2, m3 = st.columns(3)
    m1.metric("Delayed months", result["delayed_months"])
    m2.metric("Average utilization", f"{result['avg_utilization'] * 100:.1f}%")
    m3.metric("Average payment", f"{result['avg_payment']:,.0f}")

    if result["avg_utilization"] > 1:
        st.warning(
            "Average statement balance exceeds the entered credit limit. "
            "Verify the financial values before relying on this result."
        )

    st.write("")
    trends_tab, explain_tab = st.tabs(["Payment trends", "Model explanation"])

    with trends_tab:
        chart_1, chart_2 = st.columns(2)
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

    with explain_tab:
        st.caption(
            "SHAP values indicate which features moved this prediction up or down. "
            "They do not prove causation."
        )
        if st.toggle("Generate SHAP explanation"):
            shap_fig, shap_error = local_shap_chart(result["customer_fe"])
            if shap_fig is not None:
                st.plotly_chart(
                    shap_fig,
                    use_container_width=True,
                    config={"displayModeBar": False},
                )
            else:
                st.info(shap_error)

# ============================================================
# FOOTER
# ============================================================
st.markdown(
    '<div class="footer-note">Credit Risk Intelligence · Explainable assessment · '
    "Human review required</div>",
    unsafe_allow_html=True,
)