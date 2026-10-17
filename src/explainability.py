"""SHAP explanations for credit-default predictions."""

import numpy as np
import pandas as pd


def explain_default_prediction(pipeline, model_input, top_n=10):
    """
    Return SHAP contributions for one customer.

    Positive class: 1 = Default.
    SHAP units depend on the underlying model.
    """

    if not isinstance(model_input, pd.DataFrame) or len(model_input) != 1:
        raise ValueError(
            "model_input must be a DataFrame with exactly one row."
        )

    if isinstance(top_n, bool) or not isinstance(top_n, int) or top_n < 1:
        raise ValueError("top_n must be a positive integer.")

    steps = getattr(pipeline, "named_steps", {})

    if "preprocessor" not in steps or "classifier" not in steps:
        raise ValueError(
            "Pipeline requires 'preprocessor' and 'classifier' steps."
        )

    import shap

    preprocessor = steps["preprocessor"]
    classifier = steps["classifier"]

    transformed = preprocessor.transform(model_input)

    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    transformed = np.asarray(transformed)

    feature_names = list(preprocessor.get_feature_names_out())

    if len(feature_names) != transformed.shape[1]:
        raise ValueError("Feature names do not match model input.")

    explainer = shap.TreeExplainer(classifier)
    explanation = explainer(transformed)
    values = np.asarray(explanation.values)

    if values.ndim == 3 and values.shape[2] == 2:
        impacts = values[0, :, 1]
    elif values.ndim == 2:
        impacts = values[0, :]
    else:
        raise ValueError(
            f"Unsupported SHAP output shape: {values.shape}"
        )

    if len(impacts) != len(feature_names):
        raise ValueError("SHAP feature count mismatch.")

    result = pd.DataFrame({
        "feature": [
            str(name).replace("num__", "").replace("cat__", "")
            for name in feature_names
        ],
        "shap_value": impacts.astype(float)
    })

    result["absolute_impact"] = result["shap_value"].abs()

    return (
        result.sort_values("absolute_impact", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )


if __name__ == "__main__":
    import json
    import joblib
    from pathlib import Path
    from src.preprocessing import prepare_model_input

    root = Path(__file__).resolve().parent.parent

    artifact = joblib.load(
        root / "models" / "credit_default_model_v1.joblib"
    )

    with open(
        root / "config" / "input_features.json",
        encoding="utf-8"
    ) as f:
        raw_features = json.load(f)

    # Fabricated customer for technical testing
    sample = {column: 0 for column in raw_features}

    sample.update({
        "LIMIT_BAL": 200000,
        "SEX": 2,
        "EDUCATION": 2,
        "MARRIAGE": 1,
        "AGE": 30
    })

    X = prepare_model_input(
        pd.DataFrame([sample]),
        artifact["features"]
    )

    result = explain_default_prediction(
        artifact["pipeline"], X
    )

    print("SHAP EXPLANATION - SYNTHETIC CUSTOMER")
    print(result.to_string(index=False))
