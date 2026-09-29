import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sys
from pathlib import Path

# Allow imports from project root
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from src.feature_engineering import add_credit_features


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Batch Credit Risk Assessment",
    page_icon="📊",
    layout="wide"
)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():
    return joblib.load(
        ROOT / "models" / "credit_default_model_v1.joblib"
    )


artifact = load_model()

model = artifact["pipeline"]
threshold = artifact["threshold"]
model_features = artifact["features"]


# =========================================================
# REQUIRED RAW INPUT FEATURES
# =========================================================

required_features = [
    "LIMIT_BAL",
    "SEX",
    "EDUCATION",
    "MARRIAGE",
    "AGE",
    "PAY_0",
    "PAY_2",
    "PAY_3",
    "PAY_4",
    "PAY_5",
    "PAY_6",
    "BILL_AMT1",
    "BILL_AMT2",
    "BILL_AMT3",
    "BILL_AMT4",
    "BILL_AMT5",
    "BILL_AMT6",
    "PAY_AMT1",
    "PAY_AMT2",
    "PAY_AMT3",
    "PAY_AMT4",
    "PAY_AMT5",
    "PAY_AMT6"
]


# =========================================================
# HEADER
# =========================================================

st.title("Batch Credit Risk Assessment")

st.write(
    "Upload a CSV file containing multiple customers. "
    "The system will estimate default probability for each customer."
)

st.info(
    "The CSV must contain the 23 raw model input columns. "
    "Engineered features are calculated automatically."
)


# =========================================================
# EXPECTED FORMAT
# =========================================================

with st.expander("View Required CSV Columns"):

    required_df = pd.DataFrame({
        "Required Column": required_features
    })

    st.dataframe(
        required_df,
        hide_index=True,
        use_container_width=True
    )


# =========================================================
# CSV UPLOAD
# =========================================================

uploaded_file = st.file_uploader(
    "Upload Customer CSV",
    type=["csv"]
)


if uploaded_file is not None:

    try:
        data = pd.read_csv(uploaded_file)

    except Exception as error:

        st.error(
            f"Unable to read CSV file: {error}"
        )

        st.stop()


    # =====================================================
    # BASIC FILE INFORMATION
    # =====================================================

    st.subheader("Uploaded Data")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Customers",
        len(data)
    )

    col2.metric(
        "Columns",
        len(data.columns)
    )

    col3.metric(
        "Required Inputs",
        len(required_features)
    )

    st.dataframe(
        data.head(10),
        use_container_width=True
    )


    # =====================================================
    # COLUMN VALIDATION
    # =====================================================

    missing_columns = [
        column
        for column in required_features
        if column not in data.columns
    ]


    if missing_columns:

        st.error(
            "The uploaded CSV is missing required columns:"
        )

        st.write(
            ", ".join(missing_columns)
        )

        st.stop()


    # Keep model inputs only.
    # Extra columns such as Customer_ID are preserved separately.
    raw_inputs = data[
        required_features
    ].copy()


    # =====================================================
    # MISSING VALUE VALIDATION
    # =====================================================

    if raw_inputs.isnull().any().any():

        missing_count = int(
            raw_inputs.isnull().sum().sum()
        )

        st.error(
            f"The uploaded data contains "
            f"{missing_count} missing input values."
        )

        st.stop()


    # =====================================================
    # NUMERIC CONVERSION
    # =====================================================

    try:

        for column in required_features:
            raw_inputs[column] = pd.to_numeric(
                raw_inputs[column],
                errors="raise"
            )

    except Exception:

        st.error(
            "All required model input columns must contain "
            "valid numeric values."
        )

        st.stop()


    # =====================================================
    # DOMAIN VALIDATION
    # =====================================================

    errors = []


    if (raw_inputs["LIMIT_BAL"] <= 0).any():

        errors.append(
            "LIMIT_BAL must be greater than zero."
        )


    if (
        (raw_inputs["AGE"] < 18) |
        (raw_inputs["AGE"] > 100)
    ).any():

        errors.append(
            "AGE must be between 18 and 100."
        )


    pay_status_columns = [
        "PAY_0",
        "PAY_2",
        "PAY_3",
        "PAY_4",
        "PAY_5",
        "PAY_6"
    ]


    invalid_status = (
        (raw_inputs[pay_status_columns] < -2) |
        (raw_inputs[pay_status_columns] > 8)
    ).any().any()


    if invalid_status:

        errors.append(
            "Repayment status values must be "
            "between -2 and 8."
        )


    payment_columns = [
        "PAY_AMT1",
        "PAY_AMT2",
        "PAY_AMT3",
        "PAY_AMT4",
        "PAY_AMT5",
        "PAY_AMT6"
    ]


    if (
        raw_inputs[payment_columns] < 0
    ).any().any():

        errors.append(
            "Payment amounts cannot be negative."
        )


    if errors:

        st.error(
            "Input validation failed."
        )

        for error in errors:
            st.write(f"- {error}")

        st.stop()


    # =====================================================
    # FEATURE ENGINEERING
    # =====================================================

    try:

        engineered_data = add_credit_features(
            raw_inputs
        )

        engineered_data = engineered_data[
            model_features
        ]

    except Exception as error:

        st.error(
            f"Feature engineering failed: {error}"
        )

        st.stop()


    # =====================================================
    # FINAL NUMERIC VALIDATION
    # =====================================================

    numeric_values = (
        engineered_data
        .select_dtypes(include=np.number)
        .to_numpy()
    )


    if not np.isfinite(
        numeric_values
    ).all():

        st.error(
            "Engineered data contains invalid "
            "or infinite numeric values."
        )

        st.stop()


    # =====================================================
    # BATCH PREDICTION
    # =====================================================

    try:

        probabilities = model.predict_proba(
            engineered_data
        )[:, 1]

    except Exception as error:

        st.error(
            f"Prediction failed: {error}"
        )

        st.stop()


    predictions = (
        probabilities >= threshold
    ).astype(int)


    # =====================================================
    # RESULTS
    # =====================================================

    results = data.copy()

    results["Default_Probability"] = (
        probabilities
    )

    results["Default_Probability_Percent"] = (
        probabilities * 100
    ).round(2)

    results["Risk_Classification"] = np.where(
        predictions == 1,
        "Higher Risk",
        "Lower Risk"
    )

    results["Delayed_Months"] = (
        engineered_data[
            "DELAY_MONTHS"
        ].astype(int).values
    )

    results["Average_Utilization_Percent"] = (
        engineered_data[
            "AVG_UTILIZATION"
        ].values * 100
    ).round(2)


    # =====================================================
    # SUMMARY
    # =====================================================

    st.divider()

    st.subheader("Batch Assessment Summary")


    total_customers = len(results)

    higher_risk = int(
        (predictions == 1).sum()
    )

    lower_risk = int(
        (predictions == 0).sum()
    )

    average_probability = float(
        probabilities.mean() * 100
    )


    metric1, metric2, metric3, metric4 = (
        st.columns(4)
    )


    metric1.metric(
        "Total Customers",
        total_customers
    )

    metric2.metric(
        "Higher Risk",
        higher_risk
    )

    metric3.metric(
        "Lower Risk",
        lower_risk
    )

    metric4.metric(
        "Average Default Probability",
        f"{average_probability:.2f}%"
    )


    # =====================================================
    # RESULTS TABLE
    # =====================================================

    st.subheader("Customer Risk Results")


    display_columns = []

    # Preserve a likely ID column if supplied.
    for possible_id in [
        "Customer_ID",
        "CUSTOMER_ID",
        "ID",
        "customer_id"
    ]:
        if possible_id in results.columns:
            display_columns.append(
                possible_id
            )
            break


    display_columns += [
        "LIMIT_BAL",
        "AGE",
        "Default_Probability_Percent",
        "Risk_Classification",
        "Delayed_Months",
        "Average_Utilization_Percent"
    ]


    st.dataframe(
        results[display_columns],
        hide_index=True,
        use_container_width=True
    )


    # =====================================================
    # DOWNLOAD
    # =====================================================

    csv_output = results.to_csv(
        index=False
    ).encode("utf-8")


    st.download_button(
        label="Download Assessment Results",
        data=csv_output,
        file_name="credit_risk_batch_results.csv",
        mime="text/csv",
        use_container_width=True
    )


    st.success(
        "Batch assessment completed successfully."
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Predictions are machine-learning risk estimates for "
    "educational and analytical purposes. They should not "
    "be used as the sole basis for lending decisions."
)
