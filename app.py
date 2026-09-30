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
AMBER = "#FFB020"
INK = "#0E1330"

# ============================================================
# STYLE
# ============================================================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Sora:wght@600;700;800&display=swap');

    :root {
        --ink: #0E1330; --indigo: #7C3AED; --violet: #A855F7; --aqua: #19D3C5;
        --coral: #FF5C7A; --mint: #22C58B; --amber: #FFB020;
        --text: #1B2140; --muted: #6B7391; --line: #E7E1F7;
        --surface: #FFFFFF; --bg: #F5F1FF;
        --shadow-lg: 0 40px 80px -28px rgba(76,29,149,.38);
        --shadow-md: 0 20px 44px -20px rgba(76,29,149,.26);
        --shadow-sm: 0 8px 22px -12px rgba(76,29,149,.22);
        --font-head: 'Sora','Inter',sans-serif;
        --font-body: 'Inter',system-ui,sans-serif;
    }
    html {scroll-behavior: smooth;}
    html, body, .stApp, [class*="css"] {font-family: var(--font-body);}

    .stApp {
        background:
            radial-gradient(rgba(124,58,237,.10) 1px, transparent 1px) 0 0 / 26px 26px,
            radial-gradient(60rem 30rem at 0% -5%, rgba(124,58,237,.18), transparent 60%),
            radial-gradient(50rem 28rem at 100% 0%, rgba(25,211,197,.16), transparent 60%),
            radial-gradient(40rem 26rem at 50% 110%, rgba(168,85,247,.14), transparent 60%),
            var(--bg);
        color: var(--text);
    }
    .block-container {max-width: 1180px; padding-top: 1.2rem; padding-bottom: 3rem;}
    #MainMenu, footer, header[data-testid="stHeader"] {visibility: hidden;}

    /* ---------- Hero ---------- */
    .hero {
        position: relative; overflow: hidden; border-radius: 32px;
        padding: 56px 52px 50px; color: #fff; isolation: isolate;
        background:
            radial-gradient(34rem 20rem at 88% 8%, rgba(168,85,247,.65), transparent 65%),
            radial-gradient(28rem 18rem at 70% 110%, rgba(25,211,197,.45), transparent 65%),
            radial-gradient(24rem 16rem at 0% 100%, rgba(124,58,237,.6), transparent 65%),
            linear-gradient(135deg, #120826 0%, #2A1257 60%, #3B1B7A 100%);
        box-shadow: var(--shadow-lg); margin-bottom: 1.6rem;
        border: 1px solid rgba(255,255,255,.12);
    }
    .hero::before {
        content: ""; position: absolute; inset: 0; z-index: -1;
        background-image:
            linear-gradient(rgba(255,255,255,.055) 1px, transparent 1px),
            linear-gradient(90deg, rgba(255,255,255,.055) 1px, transparent 1px);
        background-size: 44px 44px;
        mask-image: radial-gradient(ellipse at 75% 30%, #000 0%, transparent 70%);
        -webkit-mask-image: radial-gradient(ellipse at 75% 30%, #000 0%, transparent 70%);
    }
    .hero-badge {
        display: inline-flex; align-items: center; gap: .55rem;
        padding: .45rem .95rem; border-radius: 999px;
        background: rgba(255,255,255,.10); border: 1px solid rgba(255,255,255,.22);
        backdrop-filter: blur(10px); color: #EDEBFF;
        font-size: .78rem; font-weight: 600; letter-spacing: .04em;
        text-transform: uppercase; margin-bottom: 1.3rem;
    }
    .hero-dot {
        width: 8px; height: 8px; border-radius: 50%; background: var(--aqua);
        box-shadow: 0 0 0 0 rgba(25,211,197,.7); animation: pulse 2.2s infinite;
    }
    @keyframes pulse {
        0% {box-shadow: 0 0 0 0 rgba(25,211,197,.65);}
        70% {box-shadow: 0 0 0 10px rgba(25,211,197,0);}
        100% {box-shadow: 0 0 0 0 rgba(25,211,197,0);}
    }
    @keyframes float {
        0%,100% {transform: translateY(-50%) rotate(-6deg);}
        50% {transform: translateY(-56%) rotate(-4deg);}
    }
    .hero-title {
        font-family: var(--font-head); font-size: clamp(2.3rem, 5vw, 3.9rem);
        font-weight: 800; letter-spacing: -.045em; line-height: 1.04;
        margin: 0; padding: 0; max-width: 640px;
        background: linear-gradient(100deg, #FFFFFF 30%, #E2CCFF 70%, #8CF3E8 100%);
        -webkit-background-clip: text; background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-copy {color: #C9CFF0; font-size: 1.05rem; margin: 1.1rem 0 0; max-width: 540px; line-height: 1.7;}

    .hero-card {
        position: absolute; right: 52px; top: 50%; width: 310px; height: 190px;
        border-radius: 24px; padding: 22px; transform: translateY(-50%) rotate(-6deg);
        background: linear-gradient(135deg, rgba(255,255,255,.26), rgba(255,255,255,.05));
        border: 1px solid rgba(255,255,255,.3); backdrop-filter: blur(18px);
        box-shadow: 0 30px 60px -20px rgba(0,0,0,.5);
        animation: float 6s ease-in-out infinite;
        display: flex; flex-direction: column; justify-content: space-between;
    }
    .hc-chip {width: 40px; height: 30px; border-radius: 7px; background: linear-gradient(135deg, #FFE08A, #E9A93B);}
    .hc-num {font-family: var(--font-head); letter-spacing: .18em; font-size: 1.05rem; color: #F3EEFF;}
    .hc-row {display: flex; justify-content: space-between; align-items: center; font-size: .72rem; color: #D7CBFF; letter-spacing: .08em;}
    .hc-pill {padding: .28rem .7rem; border-radius: 999px; background: rgba(25,211,197,.22); border: 1px solid rgba(25,211,197,.5); color: #A9F5EC; font-weight: 700;}
    @media (max-width: 980px) {.hero-card {display: none;}}

    /* ---------- Section headings ---------- */
    .section-wrap {display: flex; align-items: flex-start; gap: 1rem; margin: .2rem 0 1.1rem;}
    .section-step {
        flex: 0 0 auto; width: 40px; height: 40px; display: grid; place-items: center;
        border-radius: 13px; color: #fff; font-family: var(--font-head);
        font-weight: 700; font-size: .95rem;
        background: linear-gradient(135deg, #6D28D9, var(--violet));
        box-shadow: 0 12px 22px -8px rgba(124,58,237,.7);
    }
    .section-title {font-family: var(--font-head); color: var(--text); font-size: 1.34rem; font-weight: 700; letter-spacing: -.02em; line-height: 1.2;}
    .section-copy {color: var(--muted); font-size: .9rem; margin-top: .2rem; line-height: 1.55;}

    /* ---------- Form / cards ---------- */
    div[data-testid="stForm"] {
        position: relative; overflow: hidden;
        background: rgba(255,255,255,.88); border: 1px solid #fff;
        outline: 1px solid var(--line); border-radius: 28px;
        padding: 1.8rem 1.9rem 1.6rem; box-shadow: var(--shadow-md);
        backdrop-filter: blur(16px);
    }
    div[data-testid="stForm"]::before {
        content: ""; position: absolute; left: 0; right: 0; top: 0; height: 4px;
        background: linear-gradient(90deg, var(--indigo), var(--violet), var(--aqua));
    }
    div[data-testid="stExpander"] {
        background: rgba(255,255,255,.82); border: 1px solid var(--line);
        border-radius: 18px; overflow: hidden; box-shadow: var(--shadow-sm); margin-bottom: 1.2rem;
    }
    div[data-testid="stExpander"] details summary p {font-weight: 600; color: var(--text);}

    div[data-testid="stWidgetLabel"] p, div[data-testid="stWidgetLabel"] label, div[data-testid="stForm"] label p {
        color: var(--text) !important; font-weight: 600 !important; font-size: .88rem !important;
    }
    div[data-testid="stNumberInput"] div[data-baseweb="input"],
    div[data-testid="stNumberInput"] div[data-baseweb="base-input"],
    div[data-testid="stNumberInput"] input,
    div[data-testid="stSelectbox"] > div > div {
        background: #FBFAFF !important; background-color: #FBFAFF !important;
        border-radius: 12px !important; border-color: var(--line) !important;
    }
    div[data-testid="stNumberInput"] div[data-baseweb="input"] {border: 1px solid var(--line) !important; overflow: hidden;}
    div[data-testid="stNumberInput"] input {border: 0 !important; box-shadow: none !important;}
    div[data-testid="stNumberInput"] input,
    div[data-testid="stSelectbox"] div[data-baseweb="select"] *,
    div[data-testid="stSelectbox"] input {color: var(--text) !important; -webkit-text-fill-color: var(--text) !important;}
    div[data-testid="stNumberInput"] button {background: #EFEBFB !important; color: var(--text) !important; border: 0 !important;}
    div[data-testid="stSelectbox"] svg {fill: var(--muted) !important;}
    div[data-testid="stNumberInput"] div[data-baseweb="input"]:focus-within,
    div[data-testid="stSelectbox"] > div > div:focus-within {
        border-color: var(--indigo) !important; box-shadow: 0 0 0 3px rgba(124,58,237,.18) !important;
    }
    div[data-testid="stDataEditor"] {border: 1px solid var(--line); border-radius: 18px; overflow: hidden; box-shadow: var(--shadow-sm);}

    div[data-testid="stFormSubmitButton"] button[kind="primary"] {
        background: linear-gradient(95deg, #6D28D9 0%, var(--violet) 55%, #C084FC 100%);
        color: #fff; border: 0; border-radius: 16px; min-height: 58px;
        font-family: var(--font-head); font-weight: 700; font-size: 1.02rem;
        box-shadow: 0 22px 40px -16px rgba(124,58,237,.8);
        transition: transform .18s ease, box-shadow .18s ease, filter .18s ease;
    }
    div[data-testid="stFormSubmitButton"] button[kind="primary"]:hover {
        transform: translateY(-2px); filter: brightness(1.07);
        box-shadow: 0 26px 44px -16px rgba(124,58,237,.9);
    }
    div[data-testid="stFormSubmitButton"] button[kind="primary"]:focus-visible,
    .stButton > button:focus-visible {outline: 3px solid rgba(124,58,237,.4); outline-offset: 2px;}
    .stButton > button, div[data-testid="stDownloadButton"] > button {
        border-radius: 14px; border: 1px solid var(--line); background: #fff;
        font-weight: 600; color: var(--text); box-shadow: var(--shadow-sm);
        transition: border-color .15s ease, color .15s ease, transform .15s ease;
    }
    .stButton > button:hover, div[data-testid="stDownloadButton"] > button:hover {
        border-color: var(--indigo); color: var(--indigo); transform: translateY(-1px);
    }

    /* ---------- Result ---------- */
    .result-shell {
        position: relative; overflow: hidden; background: var(--surface);
        border: 1px solid var(--line); border-top: 5px solid var(--mint);
        border-radius: 28px; padding: 30px 32px; box-shadow: var(--shadow-lg);
        display: flex; flex-direction: column; justify-content: center;
    }
    .result-shell.mid {border-top-color: var(--amber);}
    .result-shell.high {border-top-color: var(--coral);}
    .result-shell::before {
        content: ""; position: absolute; width: 320px; height: 320px; right: -120px; top: -140px;
        border-radius: 50%; opacity: .2; filter: blur(6px);
    }
    .result-shell.low::before {background: radial-gradient(circle, var(--mint), transparent 70%);}
    .result-shell.mid::before {background: radial-gradient(circle, var(--amber), transparent 70%);}
    .result-shell.high::before {background: radial-gradient(circle, var(--coral), transparent 70%);}
    .result-top {display: flex; align-items: center; justify-content: space-between; gap: .8rem; flex-wrap: wrap;}
    .result-eyebrow {color: var(--muted); font-size: .86rem; font-weight: 600;}
    .result-status {display: inline-flex; align-items: center; gap: .45rem; border-radius: 999px; padding: .4rem .85rem; font-size: .8rem; font-weight: 700;}
    .result-shell.low .result-status {color: #0A7A54; background: #E6F8F0; border: 1px solid #C4EEDB;}
    .result-shell.mid .result-status {color: #9A6200; background: #FFF5DD; border: 1px solid #FFE2A3;}
    .result-shell.high .result-status {color: #C2264A; background: #FFEDF1; border: 1px solid #FFCFDA;}
    .result-number {
        font-family: var(--font-head); font-size: clamp(3.8rem, 7vw, 5.8rem);
        font-weight: 800; letter-spacing: -.06em; line-height: 1; margin: .7rem 0 1.2rem;
        -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent;
    }
    .result-shell.low .result-number {background-image: linear-gradient(120deg, #0E9F6E, #19D3C5);}
    .result-shell.mid .result-number {background-image: linear-gradient(120deg, #FFB020, #FF8A5C);}
    .result-shell.high .result-number {background-image: linear-gradient(120deg, #FF5C7A, #FF8A5C);}
    .meter {position: relative; height: 14px; border-radius: 999px; background: #EEEAFB; overflow: visible;}
    .meter-fill {height: 100%; border-radius: 999px;}
    .result-shell.low .meter-fill {background: linear-gradient(90deg, #22C58B, #19D3C5);}
    .result-shell.mid .meter-fill {background: linear-gradient(90deg, #FFD166, #FFB020);}
    .result-shell.high .meter-fill {background: linear-gradient(90deg, #FF8A5C, #FF5C7A);}
    .meter-mark {position: absolute; top: -5px; width: 3px; height: 24px; border-radius: 3px; background: var(--ink); transform: translateX(-50%);}
    .meter-scale {display: flex; justify-content: space-between; color: var(--muted); font-size: .76rem; font-weight: 500; margin-top: .55rem;}
    .advice {
        background: linear-gradient(120deg, #140A30, #3B1B7A); color: #F1E8FF;
        border-radius: 20px; padding: 1.05rem 1.25rem; font-size: .92rem;
        line-height: 1.65; margin-top: 1rem; box-shadow: var(--shadow-md);
        border: 1px solid rgba(255,255,255,.1);
    }
    .insight {
        background: #fff; border: 1px solid var(--line); border-left: 4px solid var(--indigo);
        border-radius: 14px; padding: .85rem 1.05rem; margin-bottom: .6rem;
        font-size: .92rem; color: var(--text); box-shadow: var(--shadow-sm);
    }

    /* ---------- Metrics ---------- */
    div[data-testid="stMetric"] {
        background: var(--surface); border: 1px solid var(--line); border-radius: 20px;
        padding: 20px 22px; box-shadow: var(--shadow-sm); position: relative; overflow: hidden;
        transition: transform .2s ease, box-shadow .2s ease;
    }
    div[data-testid="stMetric"]:hover {transform: translateY(-3px); box-shadow: var(--shadow-md);}
    div[data-testid="stMetric"]::before {
        content: ""; position: absolute; left: 0; top: 16px; bottom: 16px; width: 4px;
        border-radius: 0 4px 4px 0; background: linear-gradient(180deg, var(--indigo), var(--aqua));
    }
    div[data-testid="stMetricLabel"],
    div[data-testid="stMetricLabel"] *,
    div[data-testid="stMetricLabel"] p {
        color: #4A3F73 !important;
        -webkit-text-fill-color: #4A3F73 !important;
        font-weight: 700 !important;
        font-size: .82rem !important;
        letter-spacing: .02em;
        opacity: 1 !important;
    }
    div[data-testid="stMetricValue"] {font-family: var(--font-head); color: var(--text); font-weight: 700; letter-spacing: -.03em;}

    /* ---------- Tabs / charts ---------- */
    div[data-baseweb="tab-list"] {gap: .4rem; background: #EAE5FA; padding: .3rem; border-radius: 16px; width: fit-content;}
    button[data-baseweb="tab"] {font-weight: 600; border-radius: 12px; padding: .5rem 1.2rem; height: auto;}
    button[data-baseweb="tab"][aria-selected="true"] {background: #fff; color: var(--indigo); box-shadow: var(--shadow-sm);}
    div[data-baseweb="tab-highlight"], div[data-baseweb="tab-border"] {display: none;}
    div[data-testid="stPlotlyChart"] {background: #fff; border: 1px solid var(--line); border-radius: 22px; padding: .5rem; box-shadow: var(--shadow-sm);}

    .footer-note {color: #8A91AD; font-size: .8rem; text-align: center; padding: 2rem 0 .2rem;}
    hr {border-color: var(--line) !important; margin: 1.4rem 0 !important;}

    @media (max-width: 700px) {
        .block-container {padding-left: .9rem; padding-right: .9rem; padding-top: .8rem;}
        .hero {padding: 32px 22px 30px; border-radius: 24px;}
        div[data-testid="stForm"] {padding: 1rem .9rem 1.1rem; border-radius: 22px;}
        .result-shell {padding: 22px; border-radius: 22px;}
    }
    @media (prefers-reduced-motion: reduce) {
        .hero-dot, .hero-card {animation: none;}
        * {transition: none !important;}
    }
    /* ---------- Readability + motion polish ---------- */
    div[data-testid="stCaptionContainer"],
    div[data-testid="stCaptionContainer"] * {color: #5B5483 !important; font-size: .86rem !important;}
    button[data-baseweb="tab"] p {color: #4A3F73 !important; font-weight: 600 !important;}
    button[data-baseweb="tab"][aria-selected="true"] p {color: var(--indigo) !important;}
    div[data-testid="stExpander"] details summary p {color: var(--text) !important;}
    @keyframes rise {from {opacity: 0; transform: translateY(14px);} to {opacity: 1; transform: none;}}
    .result-shell, .advice, .insight, div[data-testid="stMetric"], div[data-testid="stPlotlyChart"] {
        animation: rise .6s ease backwards;
    }
    .scenario-note {
        background: linear-gradient(90deg, rgba(124,58,237,.10), rgba(25,211,197,.08));
        border: 1px solid rgba(124,58,237,.2); border-radius: 16px;
        padding: .9rem 1.1rem; color: #3B3360; font-size: .92rem; line-height: 1.6; margin-top: .8rem;
    }
    @media (prefers-reduced-motion: reduce) {
        .result-shell, .advice, .insight, div[data-testid="stMetric"], div[data-testid="stPlotlyChart"] {animation: none;}
    }

    /* ---------- Metric headings (robust) + primary buttons + hero CTA ---------- */
    div[data-testid="stMetric"] label,
    div[data-testid="stMetric"] label *,
    div[data-testid="stMetric"] > div:first-child,
    div[data-testid="stMetric"] > div:first-child *,
    div[data-testid="stMetricLabel"],
    div[data-testid="stMetricLabel"] * {
        color: #4A3F73 !important;
        -webkit-text-fill-color: #4A3F73 !important;
        opacity: 1 !important;
        font-weight: 700 !important;
        font-size: .82rem !important;
    }
    div[data-testid="stMetricValue"],
    div[data-testid="stMetricValue"] * {
        color: #1B2140 !important;
        -webkit-text-fill-color: #1B2140 !important;
        font-size: 2rem !important;
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(95deg, #6D28D9 0%, var(--violet) 60%, #C084FC 100%);
        color: #fff; border: 0; min-height: 52px; font-family: var(--font-head);
        box-shadow: 0 18px 34px -16px rgba(124,58,237,.8);
    }
    .stButton > button[kind="primary"]:hover {color: #fff; filter: brightness(1.07); transform: translateY(-2px);}
    .hero-cta {
        display: inline-block; margin-top: 1.5rem; padding: .7rem 1.3rem; border-radius: 14px;
        background: rgba(255,255,255,.14); border: 1px solid rgba(255,255,255,.32);
        color: #fff !important; font-weight: 600; font-size: .9rem; text-decoration: none !important;
        backdrop-filter: blur(10px); transition: background .2s ease, transform .2s ease;
    }
    .hero-cta:hover {background: rgba(255,255,255,.26); transform: translateY(-2px);}
    div[data-testid="stFileUploader"] section {
        background: #FBFAFF; border: 2px dashed #CDBDF5; border-radius: 18px;
    }
    div[data-testid="stFileUploader"] section * {color: #3B3360 !important;}
    div[data-testid="stFileUploader"] section button {background: #fff !important; border: 1px solid var(--line) !important;}
    div[data-testid="stDataFrame"] {border: 1px solid var(--line); border-radius: 18px; overflow: hidden; box-shadow: var(--shadow-sm);}

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


def risk_band(probability: float, cutoff: float) -> Tuple[str, str]:
    """Return (label, css_class) for a probability relative to the threshold."""
    if probability >= cutoff:
        return "High", "high"
    if probability >= cutoff * 0.6:
        return "Moderate", "mid"
    return "Low", "low"


def build_insights(result: Dict[str, Any]) -> List[str]:
    """Turn the engineered metrics into plain-English takeaways."""
    notes: List[str] = []

    # Repayment behaviour
    if result["max_delay"] >= 2:
        notes.append(
            f"Repayment was delayed by up to {result['max_delay']} months."
        )
    elif result["delayed_months"] == 0:
        notes.append(
            "No delayed repayments in the six-month window."
        )

    # Balance / utilization behaviour
    if result["avg_bill"] <= 0:
        notes.append(
            "No outstanding statement balance was recorded in the six-month window."
        )
    elif result["avg_utilization"] >= 0.8:
        notes.append(
            f"High average utilization ({result['avg_utilization'] * 100:.0f}% of limit)."
        )
    elif result["avg_utilization"] <= 0.3:
        notes.append(
            f"Low average utilization ({result['avg_utilization'] * 100:.0f}% of limit)."
        )

    # Payment-to-bill behaviour
    if result["avg_bill"] <= 0:
        notes.append(
            "Payment-to-bill ratio is not applicable because there is no outstanding balance."
        )
    elif result["pay_to_bill"] < 0.3:
        notes.append(
            f"Payments cover only {result['pay_to_bill'] * 100:.0f}% of balances on average."
        )
    elif result["pay_to_bill"] >= 0.8:
        notes.append(
            f"Payments cover {min(result['pay_to_bill'], 1) * 100:.0f}%+ of balances."
        )

    return notes


def simulate_payment_scenarios(customer_data: pd.DataFrame) -> pd.DataFrame:
    """Re-score the customer with all six payments scaled up or down."""
    pay_cols = [f"PAY_AMT{i}" for i in range(1, 7)]
    labels = {0.0: "No payments", 0.5: "Half", 1.0: "Current", 1.5: "+50%", 2.0: "Double"}
    rows = []
    for factor, label in labels.items():
        scenario = customer_data.copy()
        scenario[pay_cols] = scenario[pay_cols].astype(float) * factor
        try:
            scenario_fe = add_credit_features(scenario)[model_features]
            prob = float(model.predict_proba(scenario_fe)[0, 1])
        except Exception:
            continue
        rows.append({"Scenario": label, "Factor": factor, "Probability": prob * 100})
    return pd.DataFrame(rows)


def scenario_chart(scenarios: pd.DataFrame, cutoff: float) -> go.Figure:
    """Bar chart of default probability under different payment levels."""
    colors = [
        CORAL if p / 100 >= cutoff else AMBER if p / 100 >= cutoff * 0.6 else MINT
        for p in scenarios["Probability"]
    ]
    fig = go.Figure(
        go.Bar(
            x=scenarios["Scenario"],
            y=scenarios["Probability"],
            marker=dict(color=colors, cornerradius=10),
            text=[f"{v:.1f}%" for v in scenarios["Probability"]],
            textposition="outside",
            hovertemplate="%{x}<br>Default probability: %{y:.1f}%<extra></extra>",
        )
    )
    fig.add_hline(
        y=cutoff * 100,
        line_dash="dash",
        line_color=INK,
        opacity=.55,
        annotation_text=f"Threshold {cutoff * 100:.0f}%",
        annotation_position="top left",
    )
    fig.update_layout(showlegend=False, xaxis_title=None, yaxis_title=None)
    polish_chart(fig, "Default probability vs payment level", "Probability (%)", 380)
    fig.update_yaxes(range=[0, max(100 * cutoff * 1.4, float(scenarios["Probability"].max()) * 1.25)])
    return fig


BATCH_COLUMNS = [
    "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
    "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
]


def batch_template() -> pd.DataFrame:
    """Return a two-row example CSV template for batch scoring."""
    base = {
        "LIMIT_BAL": 200000, "SEX": 1, "EDUCATION": 2, "MARRIAGE": 2, "AGE": 30,
        "PAY_0": 0, "PAY_2": 0, "PAY_3": 0, "PAY_4": 0, "PAY_5": 0, "PAY_6": 0,
    }
    for i, bill in enumerate([50000, 45000, 40000, 35000, 30000, 25000], start=1):
        base[f"BILL_AMT{i}"] = bill
    for i in range(1, 7):
        base[f"PAY_AMT{i}"] = 5000
    risky = dict(base, PAY_0=2, PAY_2=2, PAY_3=1, AGE=45, LIMIT_BAL=80000)
    return pd.DataFrame([base, risky])[BATCH_COLUMNS]


def run_batch(raw: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """Score every valid row; return (results, number_of_skipped_rows)."""
    missing = [c for c in BATCH_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))

    data = raw[BATCH_COLUMNS].apply(pd.to_numeric, errors="coerce")
    valid_mask = np.isfinite(data.to_numpy(dtype=float)).all(axis=1)
    valid = data[valid_mask].copy()
    skipped = int((~valid_mask).sum())
    if valid.empty:
        raise ValueError("No valid rows found. Check the numeric values in your file.")

    features = add_credit_features(valid)[model_features]
    probs = model.predict_proba(features)[:, 1]

    results = valid.copy()
    results.insert(0, "Default Probability (%)", np.round(probs * 100, 2))
    results.insert(1, "Risk Level", [risk_band(float(x), threshold)[0] for x in probs])
    results.insert(2, "Decision", np.where(probs >= threshold, "Flag for review", "Standard"))
    return results.reset_index(drop=True), skipped


def batch_histogram(results: pd.DataFrame, cutoff: float) -> go.Figure:
    """Histogram of predicted default probabilities across the batch."""
    fig = go.Figure(
        go.Histogram(
            x=results["Default Probability (%)"],
            nbinsx=20,
            marker=dict(color=INDIGO, line=dict(color="#FFFFFF", width=1)),
            hovertemplate="%{x}%<br>Customers: %{y}<extra></extra>",
        )
    )
    fig.add_vline(
        x=cutoff * 100, line_dash="dash", line_color=CORAL,
        annotation_text=f"Threshold {cutoff * 100:.0f}%", annotation_position="top",
    )
    fig.update_layout(showlegend=False, xaxis_title="Default probability (%)", yaxis_title=None)
    polish_chart(fig, "Risk distribution across the batch", None, 340)
    return fig


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


def risk_gauge_chart(probability: float, cutoff: float) -> go.Figure:
    """Large three-band gauge (low / moderate / high) with the threshold marked."""
    pct = probability * 100
    threshold_pct = cutoff * 100
    mid_start = threshold_pct * 0.6
    label, _ = risk_band(probability, cutoff)
    bar_color = {"Low": MINT, "Moderate": AMBER, "High": CORAL}[label]

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=pct,
            number={
                "suffix": "%",
                "font": {"size": 58, "color": INK, "family": "Sora, Inter, sans-serif"},
                "valueformat": ".1f",
            },
            title={
                "text": f"Default probability · <b>{label} risk</b>",
                "font": {"size": 16, "color": "#6B7391"},
            },
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickwidth": 0,
                    "tickfont": {"size": 11, "color": "#8A91AD"},
                },
                "bar": {"color": bar_color, "thickness": .34},
                "bgcolor": "#EEF0FA",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, mid_start], "color": "#E6F8F0"},
                    {"range": [mid_start, threshold_pct], "color": "#FFF5DD"},
                    {"range": [threshold_pct, 100], "color": "#FFEDF1"},
                ],
                "threshold": {
                    "line": {"color": INK, "width": 4},
                    "thickness": .85,
                    "value": threshold_pct,
                },
            },
        )
    )
    fig.update_layout(
        height=380,
        margin=dict(l=30, r=30, t=70, b=10),
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
        <a class="hero-cta" href="#batch-processing" target="_self">Batch processing ↓</a>
        <div class="hero-card">
            <div class="hc-row"><span>CREDIT RISK</span><span class="hc-pill">AI SCORED</span></div>
            <div><div class="hc-chip"></div></div>
            <div class="hc-num">•••• •••• •••• 4821</div>
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

    # Do not score a completely inactive / empty six-month profile.
    no_balance_activity = all(float(x) == 0 for x in bills)
    no_payment_activity = all(float(x) == 0 for x in payments)

    if no_balance_activity and no_payment_activity:
        st.warning("Insufficient account activity for a meaningful assessment.")
        st.info(
            "Enter at least one statement balance or payment from the "
            "six-month account history before running the assessment."
        )
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

    st.write("")
    section(
        "3",
        "Estimated default risk",
        "The model output comes first; supporting behavior and explanation are below.",
    )

    fill_pct = min(max(probability * 100, 0.0), 100.0)
    mark_pct = min(max(threshold * 100, 0.0), 100.0)
    level, level_class = risk_band(probability, threshold)
    advice = {
        "Low": "Profile looks healthy. Standard monitoring is sufficient.",
        "Moderate": "Approaching the decision threshold. Consider a limit review and closer monitoring.",
        "High": "Above the decision threshold. Recommend manual review before extending further credit.",
    }[level]

    gauge_col, summary_col = st.columns([1.1, 1], gap="large")

    with gauge_col:
        st.plotly_chart(
            risk_gauge_chart(probability, threshold),
            use_container_width=True,
            config={"displayModeBar": False},
        )

    with summary_col:
        st.markdown(
            f"""
            <div class="result-shell {level_class}">
                <div class="result-top">
                    <div class="result-eyebrow">Next-month default probability</div>
                    <div class="result-status">● {level} risk</div>
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
            </div>
            <div class="advice"><b>Recommendation:</b> {advice}</div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Delayed months", result["delayed_months"])
    m2.metric("Average utilization", f"{result['avg_utilization'] * 100:.1f}%")
    m3.metric("Average payment", f"{result['avg_payment']:,.0f}")
    m4.metric("Payment / bill", ("N/A" if result["avg_bill"] <= 0 else f"{result['pay_to_bill'] * 100:.0f}%"))

    if result["avg_utilization"] > 1:
        st.warning(
            "Average statement balance exceeds the entered credit limit. "
            "Verify the financial values before relying on this result."
        )

    st.write("")
    section("4", "Key takeaways", "Plain-English reading of this customer's behavior.")
    for note in build_insights(result):
        st.markdown(f'<div class="insight">{note}</div>', unsafe_allow_html=True)

    summary_csv = pd.DataFrame(
        [{
            "default_probability_pct": round(probability * 100, 2),
            "risk_level": level,
            "threshold_pct": round(threshold * 100, 1),
            "delayed_months": result["delayed_months"],
            "avg_utilization_pct": round(result["avg_utilization"] * 100, 1),
            "avg_payment": round(result["avg_payment"], 0),
        }]
    ).to_csv(index=False)
    st.download_button(
        "⬇️ Download assessment summary",
        data=summary_csv,
        file_name="credit_risk_assessment.csv",
        mime="text/csv",
    )

    st.write("")
    trends_tab, scenario_tab, explain_tab = st.tabs(
        ["Payment trends", "What-if scenarios", "Model explanation"]
    )

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

    with scenario_tab:
        st.caption(
            "The same customer is re-scored with payments scaled up or down. "
            "Repayment status and balances are held constant."
        )
        scenarios = simulate_payment_scenarios(st.session_state.customer_data)
        if scenarios.empty:
            st.info("Scenario analysis is unavailable for this profile.")
        else:
            st.plotly_chart(
                scenario_chart(scenarios, threshold),
                use_container_width=True,
                config={"displayModeBar": False},
            )
            current = scenarios.loc[scenarios["Factor"] == 1.0, "Probability"]
            best = scenarios.loc[scenarios["Probability"].idxmin()]
            if not current.empty:
                delta = float(current.iloc[0]) - float(best["Probability"])
                msg = (
                    f"Lowest modelled risk is <b>{best['Probability']:.1f}%</b> "
                    f"under the <b>{best['Scenario']}</b> scenario"
                    + (f", a change of <b>{delta:.1f} points</b> from today." if delta > 0.05
                       else ": payment level has little effect on this profile.")
                )
                st.markdown(f'<div class="scenario-note">{msg}</div>', unsafe_allow_html=True)

    with explain_tab:
        st.caption(
            "SHAP values indicate which features moved this prediction up or down. "
            "They do not prove causation."
        )
        with st.spinner("Computing explanation..."):
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
# BATCH PROCESSING
# ============================================================
st.write("")
st.markdown('<div id="batch-processing"></div>', unsafe_allow_html=True)
section(
    "B",
    "Batch processing",
    "Score many customers at once: download the template, fill it in, and upload it as a CSV.",
)

tpl_col, up_col = st.columns([1, 2], gap="large")
with tpl_col:
    st.download_button(
        "⬇️ Download CSV template",
        data=batch_template().to_csv(index=False),
        file_name="batch_template.csv",
        mime="text/csv",
        use_container_width=True,
    )
    st.caption("One row per customer. All 23 columns are required.")
with up_col:
    batch_file = st.file_uploader("Upload customer CSV", type=["csv"], key="batch_upload")

if batch_file is not None:
    if st.button("🚀 Run batch scoring", type="primary", use_container_width=True):
        try:
            with st.spinner("Scoring customers..."):
                batch_results, batch_skipped = run_batch(pd.read_csv(batch_file))
            st.session_state.batch_results = batch_results
            st.session_state.batch_skipped = batch_skipped
            st.toast("Batch scoring completed.")
        except Exception as exc:
            st.session_state.pop("batch_results", None)
            st.error(f"Batch scoring failed: {exc}")

if "batch_results" in st.session_state:
    batch_out = st.session_state.batch_results
    flagged = int((batch_out["Decision"] == "Flag for review").sum())

    st.write("")
    b1, b2, b3 = st.columns(3)
    b1.metric("Customers scored", f"{len(batch_out):,}")
    b2.metric("Flagged for review", f"{flagged:,}")
    b3.metric("Average default risk", f"{batch_out['Default Probability (%)'].mean():.1f}%")

    if st.session_state.get("batch_skipped", 0):
        st.warning(f"{st.session_state.batch_skipped} row(s) were skipped because of missing or invalid values.")

    st.plotly_chart(
        batch_histogram(batch_out, threshold),
        use_container_width=True,
        config={"displayModeBar": False},
    )
    st.dataframe(
        batch_out.sort_values("Default Probability (%)", ascending=False),
        use_container_width=True,
        hide_index=True,
    )
    st.download_button(
        "⬇️ Download scored results",
        data=batch_out.to_csv(index=False),
        file_name="batch_scored_results.csv",
        mime="text/csv",
    )

# ============================================================
# FOOTER
# ============================================================
st.markdown(
    '<div class="footer-note">Credit Risk Intelligence · Decision support only</div>',
    unsafe_allow_html=True,
)