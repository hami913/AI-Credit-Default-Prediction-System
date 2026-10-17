"""HTTP-layer tests of prediction and input validation."""
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from src.feature_engineering import add_credit_features
import api

client = TestClient(api.app)


def sample_payload():
    data = {"LIMIT_BAL": 200000, "SEX": 2, "EDUCATION": 2, "MARRIAGE": 1, "AGE": 30}
    for m in [0, 2, 3, 4, 5, 6]:
        data[f"PAY_{m}"] = 0
    for m in range(1, 7):
        data[f"BILL_AMT{m}"] = 20000
        data[f"PAY_AMT{m}"] = 5000
    return data


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is True


def test_predict_matches_saved_pipeline():
    payload = sample_payload()
    response = client.post("/predict", json=payload)
    assert response.status_code == 200, response.text
    engineered = add_credit_features(pd.DataFrame([payload]))
    model_input = engineered.loc[:, api.model_features]
    expected = float(api.pipeline.predict_proba(model_input)[0, 1])
    output = response.json()
    assert output["default_probability"] == pytest.approx(round(expected, 4))
    assert output["prediction"] == int(expected >= api.threshold)


def test_invalid_credit_limit_is_rejected():
    payload = sample_payload()
    payload["LIMIT_BAL"] = 0
    assert client.post("/predict", json=payload).status_code == 422
