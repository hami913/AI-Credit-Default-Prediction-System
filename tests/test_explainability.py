import numpy as np
import pandas as pd
import pytest

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.explainability import explain_default_prediction


@pytest.fixture(scope="module")
def sample_pipeline():
    rng = np.random.default_rng(12)

    df = pd.DataFrame({
        "a": rng.normal(size=35),
        "b": rng.normal(size=35)
    })

    y = (df["a"] + df["b"] > 0).astype(int)

    pipeline = Pipeline([
        ("preprocessor", ColumnTransformer([
            ("num", StandardScaler(), ["a", "b"])
        ])),
        ("classifier", RandomForestClassifier(
            n_estimators=12, random_state=10
        ))
    ])

    pipeline.fit(df, y)

    return pipeline, df.iloc[:1]


def test_returns_explanations(sample_pipeline):
    model, X = sample_pipeline
    result = explain_default_prediction(model, X)

    assert len(result) == 2
    assert "shap_value" in result.columns
    assert np.isfinite(result["shap_value"]).all()


def test_top_n(sample_pipeline):
    model, X = sample_pipeline
    result = explain_default_prediction(model, X, top_n=1)

    assert len(result) == 1


def test_rejects_batch(sample_pipeline):
    model, X = sample_pipeline

    with pytest.raises(ValueError):
        explain_default_prediction(
            model, pd.concat([X, X])
        )


def test_invalid_top_n(sample_pipeline):
    model, X = sample_pipeline

    with pytest.raises(ValueError):
        explain_default_prediction(model, X, top_n=0)


def test_missing_pipeline_steps(sample_pipeline):
    _, X = sample_pipeline

    with pytest.raises(ValueError):
        explain_default_prediction(object(), X)
