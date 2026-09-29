import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json

from src.feature_engineering import add_credit_features


# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Credit Default Risk Prediction",
    page_icon="💳",
    layout="wide"
)


# ==================================================
# LOAD MODEL
# ==================================================

@st.cache_resource
def load_model():
    artifact = joblib.load(
        "models/credit_default_model_v1.joblib"
    )
    return artifact


artifact = load_model()

model = artifact["pipeline"]
threshold = artifact["threshold"]
model_features = artifact["features"]


# ==================================================
# LOAD ORIGINAL INPUT FEATURES
# ==================================================

with open(
    "config/input_features.json",
    "r"
) as file:
    input_features = json.load(file)


# ==================================================
# HEADER
# ==================================================

st.title("Credit Default Risk Prediction")

st.write(
    "Machine learning system for estimating a "
    "customer's probability of credit default "
    "using credit history and repayment behaviour."
)


# ==================================================
# MODEL INFORMATION
# ==================================================

with st.expander("Model Information"):

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Model",
            artifact["model_name"]
        )

    with col2:
        st.metric(
            "Version",
            artifact["model_version"]
        )

    with col3:
        st.metric(
            "Threshold",
            f"{threshold:.2f}"
        )

    with col4:
        st.metric(
            "Input Features",
            len(input_features)
        )


# ==================================================
# CUSTOMER INPUT FORM
# ==================================================

st.divider()

st.header("Customer Credit Information")

st.write(
    "Enter the customer's credit profile and "
    "six-month repayment history."
)


with st.form("credit_risk_form"):

    # ==============================================
    # BASIC CUSTOMER INFORMATION
    # ==============================================

    st.subheader("1. Customer Profile")

    col1, col2, col3 = st.columns(3)

    with col1:

        limit_bal = st.number_input(
            "Credit Limit",
            min_value=10000.0,
            max_value=1000000.0,
            value=200000.0,
            step=10000.0
        )

        age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=30,
            step=1
        )

    with col2:

        sex = st.selectbox(
            "Sex",
            options=[1, 2],
            format_func=lambda x: {
                1: "Male",
                2: "Female"
            }[x]
        )

        education = st.selectbox(
            "Education",
            options=[1, 2, 3, 4],
            format_func=lambda x: {
                1: "Graduate School",
                2: "University",
                3: "High School",
                4: "Other"
            }[x]
        )

    with col3:

        marriage = st.selectbox(
            "Marital Status",
            options=[1, 2, 3],
            format_func=lambda x: {
                1: "Married",
                2: "Single",
                3: "Other"
            }[x]
        )


    # ==============================================
    # REPAYMENT STATUS
    # ==============================================

    st.divider()

    st.subheader("2. Repayment Status")

    st.caption(
        "Enter repayment status for each of the "
        "previous six months."
    )

    repayment_options = [
        -2, -1, 0,
        1, 2, 3, 4,
        5, 6, 7, 8
    ]

    repayment_help = (
        "-2 / -1 = no delay or special repayment status, "
        "0 = revolving/on-time status, "
        "positive values indicate payment delay."
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        pay_0 = st.selectbox(
            "PAY_0 — September",
            repayment_options,
            index=2,
            help=repayment_help
        )

        pay_2 = st.selectbox(
            "PAY_2 — August",
            repayment_options,
            index=2,
            help=repayment_help
        )

    with col2:

        pay_3 = st.selectbox(
            "PAY_3 — July",
            repayment_options,
            index=2,
            help=repayment_help
        )

        pay_4 = st.selectbox(
            "PAY_4 — June",
            repayment_options,
            index=2,
            help=repayment_help
        )

    with col3:

        pay_5 = st.selectbox(
            "PAY_5 — May",
            repayment_options,
            index=2,
            help=repayment_help
        )

        pay_6 = st.selectbox(
            "PAY_6 — April",
            repayment_options,
            index=2,
            help=repayment_help
        )


    # ==============================================
    # BILL AMOUNTS
    # ==============================================

    st.divider()

    st.subheader("3. Monthly Bill Amounts")

    st.caption(
        "Enter bill statement amounts for the "
        "previous six months."
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        bill_amt1 = st.number_input(
            "September Bill",
            value=50000.0,
            step=1000.0
        )

        bill_amt2 = st.number_input(
            "August Bill",
            value=45000.0,
            step=1000.0
        )

    with col2:

        bill_amt3 = st.number_input(
            "July Bill",
            value=40000.0,
            step=1000.0
        )

        bill_amt4 = st.number_input(
            "June Bill",
            value=35000.0,
            step=1000.0
        )

    with col3:

        bill_amt5 = st.number_input(
            "May Bill",
            value=30000.0,
            step=1000.0
        )

        bill_amt6 = st.number_input(
            "April Bill",
            value=25000.0,
            step=1000.0
        )


    # ==============================================
    # PAYMENT AMOUNTS
    # ==============================================

    st.divider()

    st.subheader("4. Monthly Payment Amounts")

    st.caption(
        "Enter the amount actually paid by the "
        "customer during each month."
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        pay_amt1 = st.number_input(
            "September Payment",
            min_value=0.0,
            value=5000.0,
            step=1000.0
        )

        pay_amt2 = st.number_input(
            "August Payment",
            min_value=0.0,
            value=5000.0,
            step=1000.0
        )

    with col2:

        pay_amt3 = st.number_input(
            "July Payment",
            min_value=0.0,
            value=5000.0,
            step=1000.0
        )

        pay_amt4 = st.number_input(
            "June Payment",
            min_value=0.0,
            value=5000.0,
            step=1000.0
        )

    with col3:

        pay_amt5 = st.number_input(
            "May Payment",
            min_value=0.0,
            value=5000.0,
            step=1000.0
        )

        pay_amt6 = st.number_input(
            "April Payment",
            min_value=0.0,
            value=5000.0,
            step=1000.0
        )


    # ==============================================
    # SUBMIT BUTTON
    # ==============================================

    st.divider()

    submitted = st.form_submit_button(
        "Analyze Credit Risk",
        type="primary",
        use_container_width=True
    )


# ==================================================
# PREDICTION
# ==================================================

if submitted:

    # ----------------------------------------------
    # Build original 23-feature customer record
    # ----------------------------------------------

    customer_data = pd.DataFrame(
        [{
            "LIMIT_BAL": limit_bal,
            "SEX": sex,
            "EDUCATION": education,
            "MARRIAGE": marriage,
            "AGE": age,

            "PAY_0": pay_0,
            "PAY_2": pay_2,
            "PAY_3": pay_3,
            "PAY_4": pay_4,
            "PAY_5": pay_5,
            "PAY_6": pay_6,

            "BILL_AMT1": bill_amt1,
            "BILL_AMT2": bill_amt2,
            "BILL_AMT3": bill_amt3,
            "BILL_AMT4": bill_amt4,
            "BILL_AMT5": bill_amt5,
            "BILL_AMT6": bill_amt6,

            "PAY_AMT1": pay_amt1,
            "PAY_AMT2": pay_amt2,
            "PAY_AMT3": pay_amt3,
            "PAY_AMT4": pay_amt4,
            "PAY_AMT5": pay_amt5,
            "PAY_AMT6": pay_amt6
        }]
    )


    # ----------------------------------------------
    # Validate original features
    # ----------------------------------------------

    missing_features = [
        feature
        for feature in input_features
        if feature not in customer_data.columns
    ]

    if missing_features:

        st.error(
            "Missing required features: "
            + ", ".join(missing_features)
        )

        st.stop()


    # ----------------------------------------------
    # Feature Engineering
    # ----------------------------------------------

    customer_fe = add_credit_features(
        customer_data
    )


    # ----------------------------------------------
    # Ensure exact training feature order
    # ----------------------------------------------

    customer_fe = customer_fe[
        model_features
    ]


    # ----------------------------------------------
    # Validate numerical values
    # ----------------------------------------------

    if not np.isfinite(
        customer_fe.select_dtypes(
            include=np.number
        ).to_numpy()
    ).all():

        st.error(
            "Invalid numerical values were generated. "
            "Please check the customer information."
        )

        st.stop()


    # ----------------------------------------------
    # Model Prediction
    # ----------------------------------------------

    default_probability = (
        model.predict_proba(
            customer_fe
        )[0, 1]
    )

    prediction = int(
        default_probability >= threshold
    )

    non_default_probability = (
        1 - default_probability
    )


    # ==============================================
    # RESULTS
    # ==============================================

    st.divider()

    st.header("Credit Risk Assessment")

    result_col1, result_col2, result_col3 = (
        st.columns(3)
    )

    with result_col1:

        st.metric(
            "Default Probability",
            f"{default_probability * 100:.2f}%"
        )

    with result_col2:

        st.metric(
            "Non-Default Probability",
            f"{non_default_probability * 100:.2f}%"
        )

    with result_col3:

        st.metric(
            "Decision Threshold",
            f"{threshold * 100:.0f}%"
        )


    # ----------------------------------------------
    # Classification Result
    # ----------------------------------------------

    if prediction == 1:

        st.error(
            "Higher Default Risk"
        )

        st.write(
            "The model-estimated probability of "
            "default is above the configured "
            "classification threshold."
        )

    else:

        st.success(
            "Lower Default Risk"
        )

        st.write(
            "The model-estimated probability of "
            "default is below the configured "
            "classification threshold."
        )


    # ----------------------------------------------
    # Probability Progress
    # ----------------------------------------------

    st.subheader("Default Risk Probability")

    st.progress(
        float(
            min(
                max(default_probability, 0.0),
                1.0
            )
        )
    )

    st.write(
        f"Estimated probability: "
        f"**{default_probability * 100:.2f}%**"
    )


    # ----------------------------------------------
    # Engineered Risk Indicators
    # ----------------------------------------------

    st.subheader(
        "Customer Risk Indicators"
    )

    indicator_col1, indicator_col2, (
        indicator_col3
    ) = st.columns(3)

    with indicator_col1:

        st.metric(
            "Delayed Months",
            int(
                customer_fe[
                    "DELAY_MONTHS"
                ].iloc[0]
            )
        )

    with indicator_col2:

        st.metric(
            "Maximum Delay",
            f"{customer_fe['MAX_DELAY'].iloc[0]:.0f}"
        )

    with indicator_col3:

        st.metric(
            "Average Utilization",
            (
                f"{customer_fe['AVG_UTILIZATION'].iloc[0] * 100:.1f}%"
            )
        )


    # ----------------------------------------------
    # Additional Information
    # ----------------------------------------------

    with st.expander(
        "View Engineered Credit Indicators"
    ):

        indicators = pd.DataFrame({
            "Indicator": [
                "Average Utilization",
                "Maximum Utilization",
                "Average Delay",
                "Maximum Delay",
                "Delayed Months",
                "Average Bill Amount",
                "Average Payment Amount",
                "Payment-to-Bill Ratio",
                "Payment-to-Limit Ratio"
            ],

            "Value": [
                customer_fe[
                    "AVG_UTILIZATION"
                ].iloc[0],

                customer_fe[
                    "MAX_UTILIZATION"
                ].iloc[0],

                customer_fe[
                    "AVG_DELAY"
                ].iloc[0],

                customer_fe[
                    "MAX_DELAY"
                ].iloc[0],

                customer_fe[
                    "DELAY_MONTHS"
                ].iloc[0],

                customer_fe[
                    "AVG_BILL_AMT"
                ].iloc[0],

                customer_fe[
                    "AVG_PAY_AMT"
                ].iloc[0],

                customer_fe[
                    "PAY_TO_BILL_RATIO"
                ].iloc[0],

                customer_fe[
                    "PAY_TO_LIMIT_RATIO"
                ].iloc[0]
            ]
        })

        st.dataframe(
            indicators,
            use_container_width=True,
            hide_index=True
        )


# ==================================================
# DISCLAIMER
# ==================================================

st.divider()

st.caption(
    "This application provides a machine-learning "
    "risk estimate for educational and analytical "
    "purposes. The prediction should not be treated "
    "as an automatic loan approval or rejection "
    "decision."
)