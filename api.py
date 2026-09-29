from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.feature_engineering import add_credit_features


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "credit_default_model_v1.joblib"


# ============================================================
# LOAD MODEL
# ============================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

artifact = joblib.load(MODEL_PATH)

pipeline = artifact["pipeline"]
threshold = float(artifact.get("threshold", 0.50))


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="Credit Default Prediction API",
    description="REST API for Credit Default Risk Assessment",
    version="1.0"
)


# ============================================================
# INPUT SCHEMA
# ============================================================

class CustomerData(BaseModel):

    LIMIT_BAL: float = Field(gt=0)

    SEX: int = Field(ge=1, le=2)
    EDUCATION: int = Field(ge=1, le=4)
    MARRIAGE: int = Field(ge=1, le=3)

    AGE: int = Field(ge=18, le=100)

    PAY_0: int = Field(ge=-2, le=8)
    PAY_2: int = Field(ge=-2, le=8)
    PAY_3: int = Field(ge=-2, le=8)
    PAY_4: int = Field(ge=-2, le=8)
    PAY_5: int = Field(ge=-2, le=8)
    PAY_6: int = Field(ge=-2, le=8)

    BILL_AMT1: float
    BILL_AMT2: float
    BILL_AMT3: float
    BILL_AMT4: float
    BILL_AMT5: float
    BILL_AMT6: float

    PAY_AMT1: float = Field(ge=0)
    PAY_AMT2: float = Field(ge=0)
    PAY_AMT3: float = Field(ge=0)
    PAY_AMT4: float = Field(ge=0)
    PAY_AMT5: float = Field(ge=0)
    PAY_AMT6: float = Field(ge=0)


# ============================================================
# ENDPOINTS
# ============================================================

@app.get("/")
def home():
    return {
        "message": "Credit Default Prediction API is running",
        "version": "1.0",
        "status": "healthy"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": True,
        "model_name": artifact.get(
            "model_name",
            "Credit Default Random Forest"
        ),
        "threshold": threshold
    }


@app.post("/predict")
def predict(customer: CustomerData):

    try:
        # Convert API input to DataFrame
        raw_df = pd.DataFrame([customer.model_dump()])

        # Apply EXACT same feature engineering used in training
        engineered_df = add_credit_features(raw_df)

        # Predict probability of Class 1 = Default
        probability = float(
            pipeline.predict_proba(engineered_df)[0, 1]
        )

        prediction = int(probability >= threshold)

        return {
            "prediction": prediction,
            "prediction_label":
                "DEFAULT / HIGH RISK"
                if prediction == 1
                else "NO DEFAULT / LOWER RISK",

            "default_probability":
                round(probability, 4),

            "default_probability_percent":
                round(probability * 100, 2),

            "threshold": threshold,

            "model_version":
                artifact.get("model_version", "1.0")
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ============================================================
# END
# ============================================================
