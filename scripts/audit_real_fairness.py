import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from src.train import validate_training_data, TARGET
from src.feature_engineering import add_credit_features


ROOT = Path(__file__).resolve().parents[1]

DATA = Path(
    r"C:\Users\mianh\Downloads\UCI_Credit_Card.csv"
)

MODEL = (
    ROOT / "models" / "candidates"
    / "real_20261017_145037_cfaac2"
    / "credit_default_candidate.joblib"
)

if not DATA.is_file() or not MODEL.is_file():
    raise FileNotFoundError("Dataset or candidate model missing.")

with MODEL.open("rb") as f:
    model_hash_before = hashlib.file_digest(
        f, "sha256"
    ).hexdigest()

artifact = joblib.load(MODEL)

if artifact["model_name"] != "hist_gradient_boosting":
    raise RuntimeError("Unexpected candidate model.")

data = pd.read_csv(DATA)

clean, raw_features = validate_training_data(data)

_, test_df = train_test_split(
    clean,
    test_size=0.20,
    stratify=clean[TARGET],
    random_state=42
)

X_test = add_credit_features(
    test_df[raw_features]
).loc[:, artifact["features"]]

y = test_df[TARGET].to_numpy(dtype=int)

prob = artifact["pipeline"].predict_proba(
    X_test
)[:, 1]

# Check consistency with previously verified test results.
tn, fp, fn, tp = confusion_matrix(
    y, (prob >= 0.50).astype(int),
    labels=[0, 1]
).ravel()

if (int(tn), int(fp), int(fn), int(tp)) != (
    4445, 228, 852, 475
):
    raise RuntimeError(
        "Original held-out confusion matrix mismatch."
    )

sex = test_df["SEX"].astype(str).to_numpy()

age = pd.cut(
    test_df["AGE"],
    bins=[0, 29, 44, np.inf],
    labels=["AGE_29_OR_LESS", "AGE_30_44", "AGE_45_PLUS"]
).astype(str).to_numpy()

groups = {
    "SEX": sex,
    "AGE_BAND": age
}

def safe_div(numerator, denominator):
    if denominator == 0:
        return None
    return float(numerator / denominator)

results = []
gap_summaries = []
mlflow_metrics = {}

print("\nREAL-DATA FAIRNESS AUDIT")
print("=" * 85)
print("Model:", artifact["model_name"])
print("Held-out records:", len(y))
print("Features:", X_test.shape[1])

for threshold in (0.50, 0.25):

    predictions = (prob >= threshold).astype(int)

    print(f"\nTHRESHOLD: {threshold:.2f}")
    print("-" * 85)

    for dimension, values in groups.items():

        dimension_rows = []

        print(f"\nGROUP DIMENSION: {dimension}")
        print(
            f"{'Group':<20} {'N':>7} "
            f"{'TPR':>9} {'FPR':>9} "
            f"{'Precision':>11} {'Flagged':>9}"
        )

        for group in sorted(set(values)):
            mask = values == group

            actual = y[mask]
            predicted = predictions[mask]

            tn, fp, fn, tp = confusion_matrix(
                actual,
                predicted,
                labels=[0, 1]
            ).ravel()

            tpr = safe_div(tp, tp + fn)
            fpr = safe_div(fp, fp + tn)
            precision = safe_div(tp, tp + fp)

            selection_rate = float(predicted.mean())
            default_rate = float(actual.mean())

            row = {
                "threshold": threshold,
                "dimension": dimension,
                "group": group,
                "records": int(mask.sum()),
                "actual_defaults": int(actual.sum()),
                "actual_nondefaults": int(
                    len(actual) - actual.sum()
                ),
                "tp": int(tp),
                "fp": int(fp),
                "fn": int(fn),
                "tn": int(tn),
                "tpr": tpr,
                "fpr": fpr,
                "precision": precision,
                "flagged_rate": selection_rate,
                "actual_default_rate": default_rate,
                "limited_support": (
                    len(actual) < 100
                    or int(actual.sum()) < 30
                    or int(len(actual) - actual.sum()) < 30
                )
            }

            results.append(row)
            dimension_rows.append(row)

            fmt = lambda v: (
                f"{v:.4f}" if v is not None else "N/A"
            )

            print(
                f"{group:<20} {len(actual):>7} "
                f"{fmt(tpr):>9} {fmt(fpr):>9} "
                f"{fmt(precision):>11} "
                f"{selection_rate:>9.4f}"
            )

            if row["limited_support"]:
                print(
                    " WARNING: limited subgroup support:",
                    group
                )

        reliable = [
            r for r in dimension_rows
            if not r["limited_support"]
        ]

        def gap(field):
            vals = [
                r[field] for r in reliable
                if r[field] is not None
            ]
            if len(vals) < 2:
                return None
            return float(max(vals) - min(vals))

        gaps = {
            "threshold": threshold,
            "dimension": dimension,
            "reliable_groups": len(reliable),
            "tpr_gap": gap("tpr"),
            "fpr_gap": gap("fpr"),
            "selection_rate_gap": gap("flagged_rate")
        }

        gap_summaries.append(gaps)

        print("TPR gap:", gaps["tpr_gap"])
        print("FPR gap:", gaps["fpr_gap"])
        print(
            "Selection rate gap:",
            gaps["selection_rate_gap"]
        )

        prefix = (
            f"{dimension.lower()}_"
            f"t{int(threshold * 100)}"
        )

        for field in (
            "tpr_gap", "fpr_gap", "selection_rate_gap"
        ):
            if gaps[field] is not None:
                mlflow_metrics[f"{prefix}_{field}"] = gaps[field]

with MODEL.open("rb") as f:
    model_hash_after = hashlib.file_digest(
        f, "sha256"
    ).hexdigest()

if model_hash_before != model_hash_after:
    raise RuntimeError("Candidate model unexpectedly changed.")

stamp = datetime.now(timezone.utc).strftime(
    "%Y%m%d_%H%M%S_%f"
)

output = (
    ROOT / "reports"
    / f"real_fairness_audit_{stamp}.json"
)

output.parent.mkdir(parents=True, exist_ok=True)

report = {
    "analysis": "heldout_group_fairness_diagnostic",
    "model": artifact["model_name"],
    "records": len(y),
    "thresholds": [0.50, 0.25],
    "group_metrics": results,
    "group_gaps": gap_summaries,
    "candidate_sha256": model_hash_after,
    "deployment_approved": False,
    "limitations": [
        "Historical UCI dataset; not representative of all populations.",
        "SEX and AGE were evaluated as available group attributes.",
        "Group differences alone do not establish discrimination.",
        "No new threshold was selected from held-out results.",
        "No regulatory compliance conclusion is made.",
        "No model parameters were changed."
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
    "Credit_Risk_Real_Data_Fairness"
)

with mlflow.start_run(
    run_name="HGB_Heldout_SEX_AGE_Group_Audit"
):
    mlflow.log_params({
        "algorithm": artifact["model_name"],
        "heldout_records": len(y),
        "thresholds": "0.50,0.25",
        "attributes": "SEX,AGE_BAND"
    })

    if mlflow_metrics:
        mlflow.log_metrics(mlflow_metrics)

    mlflow.set_tags({
        "audit_type": "diagnostic_only",
        "model_modified": "false",
        "threshold_modified": "false",
        "regulatory_compliance": "not_assessed",
        "deployment_status": "not_approved"
    })

    mlflow.log_artifact(str(output))

print("\nFAIRNESS AUDIT COMPLETED")
print("Report:", output)
print("MLflow audit logged.")
print("Candidate model unchanged.")
print("Existing deployed model unchanged.")
