# 💳 Credit Risk & Loan Default Prediction System

An end-to-end Machine Learning web application designed to evaluate loan applicant default risk in real-time. Built with **XGBoost** and **Streamlit**, this engine converts financial metrics into actionable default probabilities, complete with interactive gauge charts and risk factor explainability drivers.

---

## 🌟 Key Features

- **Automated Default Scoring:** Uses a fine-tuned XGBoost model to instantly calculate risk probabilities.
- **Interactive Risk Meter:** Multi-tier Plotly spectrum gauge chart reflecting instant decision thresholds.
- **Risk Driver Breakdown:** Automated rule-based explainability highlighting critical flags (e.g., High DTI, Past Delinquencies, Low Credit Score).
- **Modern Fintech UI:** Fully customized dark-mode dashboard tailored for underwriting workflows.

---

## 🛠️ Tech Stack

- **Frontend / UI:** Streamlit, Custom HTML5/CSS3 (Glassmorphic Interface), Plotly
- **Machine Learning:** XGBoost, Scikit-Learn
- **Data & Serialization:** Pandas, NumPy, Joblib
- **Language & Runtime:** Python 3.10+

---

## 📂 Project Structure

```text
├── models/
│   ├── xgboost_model.pkl      # Trained XGBoost model artifact
│   └── scaler.pkl             # Feature scaling pipeline
├── app.py                     # Main Streamlit application script
├── train_model.py             # Model training & serialization script
├── requirements.txt           # Environment dependencies
└── README.md                  # Project documentation