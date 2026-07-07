import streamlit as st
import joblib
from xgboost import XGBClassifier
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
st.set_page_config(
    page_title="AI Credit Default Prediction",
    page_icon="💳",
    layout="wide"
)

@st.cache_resource
def load_model():
    model = XGBClassifier()
    model.load_model("credit_default_model(1).json")

    scaler = joblib.load("scaler(4).pkl")   # Agar file scaler(4).pkl hai to wahi naam likho

    return model, scaler

model, scaler = load_model()

#CSS

st.markdown("""
<style>

.main{
    background-color:#0E1117;
}

.big-title{
    font-size:40px;
    font-weight:bold;
    color:white;
}

.sub-title{
    color:#B8BCC8;
    font-size:18px;
}

.box{
    background:#1E1E1E;
    padding:20px;
    border-radius:15px;
    border:1px solid #333333;
}

.metric{
    text-align:center;
    color:white;
}

</style>
""", unsafe_allow_html=True)

#Header

st.markdown("<div class='big-title'>💳 AI Credit Default Prediction System</div>", unsafe_allow_html=True)

st.markdown("<div class='sub-title'>Predict whether a customer is likely to default on the next credit payment using Artificial Intelligence.</div>", unsafe_allow_html=True)

st.write("")

#DASHBOARD CARDS.......

c1,c2,c3=st.columns(3)

with c1:
    st.info("🤖 AI Model\n\nXGBoost Classifier")

with c2:
    st.success("📊 Features\n\n23 Customer Features")

with c3:
    st.warning("⚡ Accuracy\n\n84%")

#Welcome Box........

st.markdown("---")

st.markdown("""
### 👋 Welcome

Fill in the customer information.

Then click **Predict Credit Risk** to get the prediction.
""")

#👤 Section 1: Customer Information

st.markdown("---")
st.subheader("👤 Customer Information")

col1, col2, col3 = st.columns(3)

with col1:
    limit_bal = st.number_input(
        "Credit Limit (LIMIT_BAL)",
        min_value=10000,
        max_value=1000000,
        value=50000,
        step=1000
    )

with col2:
    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=30
    )

with col3:
    sex = st.selectbox(
        "Gender",
        [1, 2],
        format_func=lambda x: "Male" if x == 1 else "Female"
    )

col4, col5 = st.columns(2)

with col4:
    education = st.selectbox(
        "Education",
        [1,2,3,4],
        format_func=lambda x: {
            1:"Graduate School",
            2:"University",
            3:"High School",
            4:"Others"
        }[x]
    )

with col5:
    marriage = st.selectbox(
        "Marriage",
        [1,2,3],
        format_func=lambda x:{
            1:"Married",
            2:"Single",
            3:"Others"
        }[x]
    )

# 📅 Section 2: Payment History

st.markdown("---")
st.subheader("📅 Payment History")

status_options = [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8]

def payment_status_text(x):
    return {
        -2: "No Consumption",
        -1: "Paid Duly",
         0: "Use of Revolving Credit",
         1: "1 Month Delay",
         2: "2 Months Delay",
         3: "3 Months Delay",
         4: "4 Months Delay",
         5: "5 Months Delay",
         6: "6 Months Delay",
         7: "7 Months Delay",
         8: "8+ Months Delay"
    }[x]

col1, col2, col3 = st.columns(3)

with col1:
    pay_0 = st.selectbox(
        "September Payment Status (PAY_0)",
        status_options,
        format_func=payment_status_text
    )

with col2:
    pay_2 = st.selectbox(
        "August Payment Status (PAY_2)",
        status_options,
        format_func=payment_status_text
    )

with col3:
    pay_3 = st.selectbox(
        "July Payment Status (PAY_3)",
        status_options,
        format_func=payment_status_text
    )

col4, col5, col6 = st.columns(3)

with col4:
    pay_4 = st.selectbox(
        "June Payment Status (PAY_4)",
        status_options,
        format_func=payment_status_text
    )

with col5:
    pay_5 = st.selectbox(
        "May Payment Status (PAY_5)",
        status_options,
        format_func=payment_status_text
    )

with col6:
    pay_6 = st.selectbox(
        "April Payment Status (PAY_6)",
        status_options,
        format_func=payment_status_text
    )

    #💳 Section 3: Bill Amount Information

    st.markdown("---")
st.subheader("💳 Bill Amount Information")

col1, col2, col3 = st.columns(3)

with col1:
    bill_amt1 = st.number_input(
        "Bill Amount September (BILL_AMT1)",
        min_value=0,
        value=5000,
        step=100
    )

with col2:
    bill_amt2 = st.number_input(
        "Bill Amount August (BILL_AMT2)",
        min_value=0,
        value=4500,
        step=100
    )

with col3:
    bill_amt3 = st.number_input(
        "Bill Amount July (BILL_AMT3)",
        min_value=0,
        value=4000,
        step=100
    )

col4, col5, col6 = st.columns(3)

with col4:
    bill_amt4 = st.number_input(
        "Bill Amount June (BILL_AMT4)",
        min_value=0,
        value=3500,
        step=100
    )

with col5:
    bill_amt5 = st.number_input(
        "Bill Amount May (BILL_AMT5)",
        min_value=0,
        value=3000,
        step=100
    )

with col6:
    bill_amt6 = st.number_input(
        "Bill Amount April (BILL_AMT6)",
        min_value=0,
        value=2500,
        step=100
    )

   # 💵 Section 4: Payment Amount Information

st.markdown("---")
st.subheader("💵 Payment Amount Information")

col1, col2, col3 = st.columns(3)

with col1:
    pay_amt1 = st.number_input(
        "Payment Amount September (PAY_AMT1)",
        min_value=0,
        value=2000,
        step=100
    )

with col2:
    pay_amt2 = st.number_input(
        "Payment Amount August (PAY_AMT2)",
        min_value=0,
        value=1800,
        step=100
    )

with col3:
    pay_amt3 = st.number_input(
        "Payment Amount July (PAY_AMT3)",
        min_value=0,
        value=1600,
        step=100
    )

col4, col5, col6 = st.columns(3)

with col4:
    pay_amt4 = st.number_input(
        "Payment Amount June (PAY_AMT4)",
        min_value=0,
        value=1400,
        step=100
    )

with col5:
    pay_amt5 = st.number_input(
        "Payment Amount May (PAY_AMT5)",
        min_value=0,
        value=1200,
        step=100
    )

with col6:
    pay_amt6 = st.number_input(
        "Payment Amount April (PAY_AMT6)",
        min_value=0,
        value=1000,
        step=100
    )
# PREDICTION BUTTON
import numpy as np

st.markdown("---")

if st.button("🔍 Predict Credit Risk", use_container_width=True):

    # ⭐ ADD THIS HERE (VERY IMPORTANT)
    edu_map = {
        1: "Graduate School",
        2: "University",
        3: "High School",
        4: "Others"
    }

    mar_map = {
        1: "Married",
        2: "Single",
        3: "Others"
    }

    input_data = np.array([[
        limit_bal, sex, education, marriage, age,
        pay_0, pay_2, pay_3, pay_4, pay_5, pay_6,
        bill_amt1, bill_amt2, bill_amt3, bill_amt4, bill_amt5, bill_amt6,
        pay_amt1, pay_amt2, pay_amt3, pay_amt4, pay_amt5, pay_amt6
    ]])

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]

    risk = probability[1] * 100

    st.markdown("---")
    st.subheader("📊 AI Prediction Report")

    st.progress(int(min(risk, 100)))

    if risk < 30:
        st.success("🟢 Low Risk")
        recommendation = "Customer appears financially stable. Credit approval is recommended."

    elif risk < 70:
        st.warning("🟡 Medium Risk")
        recommendation = "Approve with caution. Consider reducing the credit limit."

    else:
        st.error("🔴 High Risk")
        recommendation = "High probability of default. Manual review is recommended."

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Default Probability", f"{risk:.2f}%")

    with col2:
        st.metric("Safe Probability", f"{100-risk:.2f}%")

    # =========================
    # 🥧 PIE CHART (INSIDE BUTTON)
    # =========================
    import plotly.express as px
    import plotly.graph_objects as go

    st.markdown("---")
    st.subheader("🥧 Prediction Probability Breakdown")

    pie_data = pd.DataFrame({
        "Category": ["Safe", "Default"],
        "Probability": [100 - risk, risk]
    })

    fig = px.pie(pie_data, names="Category", values="Probability", hole=0.5)
    st.plotly_chart(fig, use_container_width=True)

    # =========================
    # 🎯 GAUGE
    # =========================
    st.markdown("---")
    st.subheader("🎯 Risk Gauge Meter")

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk,
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkred"},
            'steps': [
                {'range': [0, 30], 'color': "green"},
                {'range': [30, 70], 'color': "yellow"},
                {'range': [70, 100], 'color': "red"}
            ]
        }
    ))

    st.plotly_chart(fig, use_container_width=True)

    # =========================
    # 🤖 RECOMMENDATION
    # =========================
    st.markdown("---")
    st.subheader("🤖 AI Recommendation")
    st.info(recommendation)

    # =========================
    # 📋 SUMMARY
    # =========================
    st.markdown("---")
    st.subheader("📋 Customer Summary")

    st.write(f"""
**Age:** {age}

**Credit Limit:** {limit_bal}

**Gender:** {"Male" if sex == 1 else "Female"}

**Education:** {edu_map[education]}

**Marriage:** {mar_map[marriage]}
""")

    # =========================
    # 📊 FEATURE IMPORTANCE
    # =========================
    st.markdown("---")
    st.subheader("📊 Feature Importance")

    feature_names = [
        "LIMIT_BAL","SEX","EDUCATION","MARRIAGE","AGE",
        "PAY_0","PAY_2","PAY_3","PAY_4","PAY_5","PAY_6",
        "BILL_AMT1","BILL_AMT2","BILL_AMT3",
        "BILL_AMT4","BILL_AMT5","BILL_AMT6",
        "PAY_AMT1","PAY_AMT2","PAY_AMT3",
        "PAY_AMT4","PAY_AMT5","PAY_AMT6"
    ]

    importance = model.feature_importances_

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importance
    }).sort_values(by="Importance", ascending=True)

    fig = px.bar(
        importance_df,
        x="Importance",
        y="Feature",
        orientation="h"
    )

    st.plotly_chart(fig, use_container_width=True)

    # =========================
    # 📥 DOWNLOAD REPORT
    # =========================
    st.markdown("---")
    st.subheader("📥 Download Prediction Report")

    report = pd.DataFrame({
        "Feature": [
            "Age","Credit Limit","Gender","Education","Marriage",
            "Risk Level","Default Probability"
        ],
        "Value": [
            age,
            limit_bal,
            "Male" if sex == 1 else "Female",
            edu_map[education],
            mar_map[marriage],
            "Low Risk" if risk < 30 else "Medium Risk" if risk < 70 else "High Risk",
            f"{risk:.2f}%"
        ]
    })

    csv = report.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="📥 Download CSV Report",
        data=csv,
        file_name="credit_prediction_report.csv",
        mime="text/csv",
        use_container_width=True
    )