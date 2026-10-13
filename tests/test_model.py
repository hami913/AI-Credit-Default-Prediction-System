from pathlib import Path
import joblib
import numpy as np
import pytest

MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "models"
    / "credit_default_model_v1.joblib"
)

@pytest.fixture(scope="module")
def artifact():
    assert MODEL_PATH.exists(), "Model file not found"
    return joblib.load(MODEL_PATH)

def test_model_loads(artifact):
    assert "pipeline" in artifact

def test_threshold_exists(artifact):
    assert "threshold" in artifact

def test_threshold_range(artifact):
    assert 0 <= artifact["threshold"] <= 1

def test_model_classes(artifact):
    model = artifact["pipeline"]
    assert np.array_equal(model.classes_, [0, 1])

def test_model_has_predict_proba(artifact):
    model = artifact["pipeline"]
    assert callable(getattr(model, "predict_proba", None))

# ==============================================
# PHASE 2: AUTOMATED PREDICTION TESTING
# ==============================================

import pandas as pd



@pytest.fixture(scope="module")
def sample_customer(artifact):
    """Create one internally consistent test customer."""

    import pandas as pd

    features = list(artifact["features"])

    customer = {
        "LIMIT_BAL": 200000,
        "SEX": 2,
        "EDUCATION": 2,
        "MARRIAGE": 1,
        "AGE": 30,
    }

    for month in [0, 2, 3, 4, 5, 6]:
        customer[f"PAY_{month}"] = 0

    for month in range(1, 7):
        customer[f"BILL_AMT{month}"] = 20000
        customer[f"PAY_AMT{month}"] = 5000

        customer[f"UTILIZATION_{month}"] = (
            customer[f"BILL_AMT{month}"]
            / customer["LIMIT_BAL"]
        )

    utilization = [
        customer[f"UTILIZATION_{i}"]
        for i in range(1, 7)
    ]

    delays = [
        customer[f"PAY_{i}"]
        for i in [0, 2, 3, 4, 5, 6]
    ]

    bills = [
        customer[f"BILL_AMT{i}"]
        for i in range(1, 7)
    ]

    payments = [
        customer[f"PAY_AMT{i}"]
        for i in range(1, 7)
    ]

    customer["AVG_UTILIZATION"] = sum(utilization) / 6
    customer["MAX_UTILIZATION"] = max(utilization)

    customer["AVG_DELAY"] = sum(delays) / 6
    customer["MAX_DELAY"] = max(delays)
    customer["DELAY_MONTHS"] = sum(d > 0 for d in delays)

    customer["AVG_BILL_AMT"] = sum(bills) / 6
    customer["AVG_PAY_AMT"] = sum(payments) / 6

    customer["PAY_TO_BILL_RATIO"] = (
        sum(payments) / max(sum(bills), 1)
    )

    customer["PAY_TO_LIMIT_RATIO"] = (
        (sum(payments) / 6) / customer["LIMIT_BAL"]
    )

    missing = set(features) - set(customer)
    assert not missing, f"Missing features: {missing}"

    return pd.DataFrame(
        [customer],
        columns=features
    )


def test_single_customer_prediction(artifact, sample_customer):
    model = artifact["pipeline"]

    prediction = model.predict(sample_customer)

    assert len(prediction) == 1
    assert prediction[0] in [0, 1]


def test_prediction_probability(artifact, sample_customer):
    model = artifact["pipeline"]

    probabilities = model.predict_proba(sample_customer)

    assert probabilities.shape == (1, 2)
    assert np.all(np.isfinite(probabilities))
    assert np.all(probabilities >= 0)
    assert np.all(probabilities <= 1)
    assert np.allclose(probabilities.sum(axis=1), 1)


def test_batch_prediction(artifact, sample_customer):
    model = artifact["pipeline"]

    batch = pd.concat(
        [sample_customer] * 5,
        ignore_index=True
    )

    predictions = model.predict(batch)

    assert len(predictions) == 5
    assert all(p in [0, 1] for p in predictions)


def test_prediction_consistency(artifact, sample_customer):
    model = artifact["pipeline"]

    first = model.predict_proba(sample_customer)
    second = model.predict_proba(sample_customer)

    assert np.allclose(first, second)

