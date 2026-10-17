# Credit Default Prediction System

Research/educational system predicting **next-month default risk for existing credit-card customers**. This is not a validated loan-approval engine and must not be used for automatic lending decisions.

## Inputs and outputs
- Input: 23 raw credit-card customer features from the UCI default-of-credit-card-clients dataset.
- Feature engineering: `src/feature_engineering.py`.
- Inference: loads `models/credit_default_model_v1.joblib` and uses its stored `features` and `threshold`.
- Output: predicted default probability and threshold-based classification.

## Local setup (PowerShell)
```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest tests/ -v
.\.venv\Scripts\python.exe -m uvicorn api:app --reload
```
Open `http://127.0.0.1:8000/docs` to try the API.

## Dashboard
```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

## Docker API
```powershell
docker build -t credit-risk-api .
docker run --rm -p 8000:8000 credit-risk-api
```

## Limits
- Data consists of historical credit card behavior and is not representative of every lender or population.
- Risk scores need calibration, fairness evaluation and external validation before any consequential use.
- Retraining, model selection, hyperparameter searches and MLflow runs are not included in the API/dashboard source folder.
