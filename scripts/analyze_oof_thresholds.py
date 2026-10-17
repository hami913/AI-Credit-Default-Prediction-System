import json
from pathlib import Path
from datetime import datetime, timezone

import joblib
import mlflow
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_predict
)

from src.train import (
    validate_training_data,
    TARGET
)
from src.feature_engineering import add_credit_features
from src.evaluate import evaluate_model


ROOT = Path(__file__).resolve().parents[1]

DATASET = Path(
    r"C:\Users\mianh\Downloads\UCI_Credit_Card.csv"
)

CANDIDATE = (
    ROOT / "models" / "candidates"
    / "real_20261017_145037_cfaac2"
    / "credit_default_candidate.joblib"
)

if not CANDIDATE.exists():
    raise FileNotFoundError(CANDIDATE)

artifact = joblib.load(CANDIDATE)

if artifact["model_name"] != "hist_gradient_boosting":
    raise ValueError("Unexpected candidate algorithm.")

data = pd.read_csv(DATASET)
clean, raw = validate_training_data(data)

# Reproduce original candidate training split.
train_df, _ = train_test_split(
    clean,
    test_size=0.20,
    stratify=clean[TARGET],
    random_state=42
)

X = add_credit_features(
    train_df[raw]
).loc[:, artifact["features"]]

y = train_df[TARGET].astype(int)

cv = StratifiedKFold(
    n_splits=3,
    shuffle=True,
    random_state=42
)

print("Generating out-of-fold probabilities...")
print("Training records:", len(X))
print("Features:", X.shape[1])

# Each prediction comes from a model that
# was not fitted on that record.
probabilities = cross_val_predict(
    clone(artifact["pipeline"]),
    X,
    y,
    cv=cv,
    method="predict_proba",
    n_jobs=1
)[:, 1]

thresholds = np.round(
    np.arange(0.20, 0.601, 0.05),
    2
)

results = []

for threshold in thresholds:
    metrics = evaluate_model(
        y.to_numpy(),
        probabilities,
        threshold=float(threshold)
    )

    results.append({
        "Threshold": float(threshold),
        "Precision": float(metrics["Precision"]),
        "Recall": float(metrics["Recall"]),
        "F1": float(metrics["F1"]),
        "FP": int(metrics["FP"]),
        "FN": int(metrics["FN"])
    })

table = pd.DataFrame(results)

print("\nOUT-OF-FOLD THRESHOLD COMPARISON")
print("=" * 75)
print(
    table.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

# Exploratory threshold maximizing OOF F1
best = max(results, key=lambda row: row["F1"])

print("\nEXPLORATORY F1-OPTIMAL THRESHOLD")
print("Threshold:", best["Threshold"])
print("Precision:", round(best["Precision"], 4))
print("Recall:", round(best["Recall"], 4))
print("F1:", round(best["F1"], 4))
print("False negatives:", best["FN"])

output_dir = ROOT / "reports"
output_dir.mkdir(exist_ok=True)

stamp = datetime.now(timezone.utc).strftime(
    "%Y%m%d_%H%M%S"
)

output = output_dir / f"oof_threshold_analysis_{stamp}.json"

report = {
    "evaluation": "training_out_of_fold",
    "model": artifact["model_name"],
    "model_features": len(artifact["features"]),
    "cv_folds": 3,
    "training_records": len(X),
    "threshold_results": results,
    "exploratory_best_f1": best,
    "deployment_approved": False,
    "limitations": [
        "Threshold selection is exploratory.",
        "Candidate hyperparameters were previously selected.",
        "Do not use held-out test data for threshold tuning.",
        "Fairness and cost validation remain necessary."
    ]
}

output.write_text(
    json.dumps(report, indent=2),
    encoding="utf-8"
)

mlflow.set_tracking_uri(
    "http://127.0.0.1:5000"
)

mlflow.set_experiment(
    "Credit_Risk_Threshold_Validation"
)

with mlflow.start_run(
    run_name="HGB_Training_OOF_Threshold_Analysis"
):
    mlflow.log_params({
        "algorithm": artifact["model_name"],
        "cv_folds": 3,
        "training_records": len(X)
    })

    mlflow.log_metrics({
        "oof_roc_auc": float(
            evaluate_model(y, probabilities)["ROC_AUC"]
        ),
        "oof_average_precision": float(
            evaluate_model(y, probabilities)["PR_AUC"]
        ),
        "exploratory_threshold": best["Threshold"],
        "exploratory_f1": best["F1"],
        "exploratory_recall": best["Recall"]
    })

    mlflow.set_tags({
        "evaluation_type": "training_oof",
        "threshold_status": "exploratory_only",
        "deployment_status": "not_approved"
    })

    mlflow.log_artifact(str(output))

print("\nReport saved:", output)
print("MLflow threshold experiment logged.")
print("Existing deployed model unchanged.")
