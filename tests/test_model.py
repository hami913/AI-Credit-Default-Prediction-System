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
