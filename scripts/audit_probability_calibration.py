import json
from pathlib import Path
from datetime import datetime, timezone

import joblib
import mlflow
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    roc_auc_score,
    average_precision_score
)

from src.train import validate_training_data, TARGET
from src.feature_engineering import add_credit_features


ROOT = Path(__file__).resolve().parents[1]

DATASET = Path(
    r"C:\Users\mianh\Downloads\UCI_Credit_Card.csv"
)

MODEL = (
    ROOT / "models" / "candidates"
    / "real_20261017_145037_cfaac2"
    / "credit_default_candidate.joblib"
)

artifact = joblib.load(MODEL)

if artifact["model_name"] != "hist_gradient_boosting":
    raise RuntimeError("Unexpected candidate model.")

# Reproduce the existing held-out split.
data = pd.read_csv(DATASET)
clean, raw_features = validate_training_data(data)

_, test_df = train_test_split(
    clean,
    test_size=0.20,
    random_state=42,
    stratify=clean[TARGET]
)

X_test = add_credit_features(
    test_df[raw_features]
).loc[:, artifact["features"]]

y_true = test_df[TARGET].to_numpy(dtype=int)

prob = artifact["pipeline"].predict_proba(
    X_test
)[:, 1]

# Verify consistency with the previous held-out report.
brier = float(brier_score_loss(y_true, prob))

if not np.isclose(
    brier, 0.13474164836481928, atol=1e-6
):
    raise RuntimeError(
        "Held-out evaluation mismatch. "
        "Calibration audit stopped."
    )

metrics = {
    "brier_score": brier,
    "log_loss": float(log_loss(y_true, prob)),
    "roc_auc": float(roc_auc_score(y_true, prob)),
    "average_precision": float(
        average_precision_score(y_true, prob)
    ),
    "mean_predicted_probability": float(prob.mean()),
    "observed_default_rate": float(y_true.mean())
}

# Fixed-width calibration bins: 0-10%, ..., 90-100%.
edges = np.linspace(0, 1, 11)

bin_ids = np.clip(
    np.searchsorted(edges, prob, side="right") - 1,
    0,
    9
)

rows = []
ece = 0.0
mce = 0.0

for i in range(10):
    mask = bin_ids == i
    count = int(mask.sum())

    if count == 0:
        continue

    predicted = float(prob[mask].mean())
    observed = float(y_true[mask].mean())
    gap = abs(predicted - observed)

    ece += (count / len(y_true)) * gap
    mce = max(mce, gap)

    rows.append({
        "bin": f"{i*10}-{(i+1)*10}%",
        "count": count,
        "mean_predicted": predicted,
        "actual_default_rate": observed,
        "absolute_gap": gap
    })

metrics["expected_calibration_error"] = float(ece)
metrics["maximum_calibration_error"] = float(mce)

print("\nPROBABILITY CALIBRATION AUDIT")
print("=" * 76)
print("Model:", artifact["model_name"])
print("Held-out customers:", len(y_true))
print("Features:", X_test.shape[1])

print("\nSUMMARY METRICS")
for key, value in metrics.items():
    print(f"{key:<35}: {value:.6f}")

print("\nCALIBRATION BY PROBABILITY BAND")
print("=" * 76)

print(
    f"{'Band':<12}"
    f"{'Customers':>12}"
    f"{'Predicted':>15}"
    f"{'Observed':>15}"
    f"{'Gap':>12}"
)

for row in rows:
    print(
        f"{row['bin']:<12}"
        f"{row['count']:>12}"
        f"{row['mean_predicted']:>15.4f}"
        f"{row['actual_default_rate']:>15.4f}"
        f"{row['absolute_gap']:>12.4f}"
    )

output_dir = ROOT / "reports"
output_dir.mkdir(parents=True, exist_ok=True)

stamp = datetime.now(timezone.utc).strftime(
    "%Y%m%d_%H%M%S_%f"
)

output = (
    output_dir
    / f"candidate_calibration_audit_{stamp}.json"
)

report = {
    "analysis_type": "heldout_calibration_diagnostic",
    "model": artifact["model_name"],
    "records": len(y_true),
    "metrics": metrics,
    "calibration_bins": rows,
    "model_retrained": False,
    "calibration_fitted": False,
    "deployment_approved": False,
    "limitations": [
        "Held-out diagnostic only.",
        "No calibration model fitted on test data.",
        "Do not tune the model against this audit.",
        "Separate validation data is needed for recalibration.",
        "Business and fairness approval still pending."
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
    "Credit_Risk_Probability_Calibration"
)

with mlflow.start_run(
    run_name="HGB_Heldout_Calibration_Audit"
):
    mlflow.log_params({
        "algorithm": artifact["model_name"],
        "evaluation_records": len(y_true),
        "calibration_bins": 10
    })

    mlflow.log_metrics(metrics)

    mlflow.set_tags({
        "evaluation_type": "heldout_diagnostic",
        "calibration_fitted": "false",
        "model_modified": "false",
        "deployment_status": "not_approved"
    })

    mlflow.log_artifact(str(output))

print("\nReport saved:", output)
print("MLflow calibration audit logged.")
print("Existing deployed model unchanged.")
