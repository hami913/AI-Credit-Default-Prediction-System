import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from src.train import validate_training_data, TARGET
from src.feature_engineering import add_credit_features
from src.evaluate import evaluate_model


ROOT = Path(__file__).resolve().parents[1]

DATASET = Path(
    r"C:\Users\mianh\Downloads\UCI_Credit_Card.csv"
)

CANDIDATE_DIR = (
    ROOT / "models" / "candidates"
    / "real_20261017_145037_cfaac2"
)

ARTIFACT = joblib.load(
    CANDIDATE_DIR / "credit_default_candidate.joblib"
)

if ARTIFACT["model_name"] != "hist_gradient_boosting":
    raise ValueError("Unexpected model.")

# Reconstruct original held-out split
data = pd.read_csv(DATASET)
clean, raw_features = validate_training_data(data)

_, test_df = train_test_split(
    clean,
    test_size=0.20,
    stratify=clean[TARGET],
    random_state=42
)

X_test = add_credit_features(
    test_df[raw_features]
).loc[:, ARTIFACT["features"]]

y_test = test_df[TARGET].to_numpy()

probabilities = ARTIFACT["pipeline"].predict_proba(
    X_test
)[:, 1]

baseline = evaluate_model(
    y_test, probabilities, threshold=0.50
)

# Verify we reproduced the previously logged baseline
expected_auc = 0.7825704624249585

if not np.isclose(
    baseline["ROC_AUC"], expected_auc, atol=1e-6
):
    raise RuntimeError(
        "Baseline ROC-AUC mismatch. Check original split."
    )

selected = evaluate_model(
    y_test, probabilities, threshold=0.25
)

print("\nHELD-OUT THRESHOLD COMPARISON")
print("=" * 70)

print(
    f"{'Metric':<20} {'Threshold 0.50':>16} {'Threshold 0.25':>16}"
)

for metric in [
    "Precision", "Recall", "F1",
    "Accuracy", "FP", "FN", "TP", "TN"
]:
    print(
        f"{metric:<20} "
        f"{baseline[metric]:>16.4f} "
        f"{selected[metric]:>16.4f}"
    )

def native(metrics):
    return {
        k: v.item() if isinstance(v, np.generic) else v
        for k, v in metrics.items()
    }

report = {
    "model": ARTIFACT["model_name"],
    "model_version": ARTIFACT["model_version"],
    "heldout_records": len(test_df),
    "selection_source": "training_oof_f1",
    "baseline": native(baseline),
    "proposed_threshold": native(selected),
    "deployment_approved": False,
    "limitations": [
        "Threshold selected using training OOF estimates.",
        "Candidate test results were previously inspected.",
        "Further test-based threshold tuning is not permitted.",
        "Business cost and fairness assessment required."
    ]
}

output = (
    ROOT / "reports"
    / (
        "candidate_threshold_validation_"
        + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        + ".json"
    )
)

output.parent.mkdir(exist_ok=True)

output.write_text(
    json.dumps(report, indent=2, allow_nan=False),
    encoding="utf-8"
)

mlflow.set_tracking_uri(
    "http://127.0.0.1:5000"
)

mlflow.set_experiment(
    "Credit_Risk_Candidate_Threshold_Check"
)

with mlflow.start_run(
    run_name="HGB_Fixed_025_Heldout_Check"
):
    mlflow.log_params({
        "model": ARTIFACT["model_name"],
        "baseline_threshold": 0.50,
        "proposed_threshold": 0.25,
        "selection_source": "training_oof_f1"
    })

    for prefix, metrics in [
        ("baseline", baseline),
        ("proposed", selected)
    ]:
        for key in ["Precision", "Recall", "F1", "Accuracy"]:
            mlflow.log_metric(
                f"{prefix}_{key.lower()}",
                float(metrics[key])
            )

    mlflow.set_tags({
        "evaluation_type": "heldout_threshold_check",
        "deployment_status": "not_approved"
    })

    mlflow.log_artifact(str(output))

print("\nReport saved:", output)
print("MLflow tracking complete.")
print("Existing deployed model unchanged.")
