"""Train and track real credit-risk candidates in MLflow."""

import hashlib
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import mlflow
import pandas as pd

from src.train import train_candidate


ROOT = Path(__file__).resolve().parents[1]

CSV = Path(
    r"C:\Users\mianh\Downloads\UCI_Credit_Card.csv"
)

URI = "http://127.0.0.1:5000"

if not CSV.is_file():
    raise FileNotFoundError(CSV)

# Verify that MLflow server is running
with urllib.request.urlopen(URI, timeout=10) as response:
    if response.status != 200:
        raise RuntimeError("MLflow server unavailable.")

# Use unique directory to protect existing model
output = ROOT / "models" / "candidates" / (
    "real_"
    + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    + "_"
    + uuid4().hex[:6]
)

print("TRAINING THREE ALGORITHMS ON REAL DATA", flush=True)
print("Existing deployed model will not be modified.", flush=True)

# Execute training, tuning and cross-validation
result = train_candidate(
    pd.read_csv(CSV),
    output,
    folds=3,
    iterations=3,
    seed=42
)

with CSV.open("rb") as f:
    dataset_hash = hashlib.file_digest(
        f, "sha256"
    ).hexdigest()

# Connect to running MLflow server
mlflow.set_tracking_uri(URI)

mlflow.set_experiment(
    "Credit_Risk_Real_Training_Comparison"
)

with mlflow.start_run(
    run_name="UCI_3_Algorithm_Candidate_Comparison"
) as parent:

    mlflow.log_params({
        "dataset_rows": result["dataset_rows"],
        "train_rows": result["train_rows"],
        "heldout_test_rows": result["heldout_test_rows"],
        "feature_count": result["feature_count"],
        "cv_folds": result["cv_folds"],
        "iterations_per_algorithm":
            result["search_iterations_per_algorithm"],
        "random_seed": 42
    })

    mlflow.set_tags({
        "run_type": "real_training_comparison",
        "training_executed": "true",
        "selection_metric": result["selection_metric"],
        "selected_algorithm": result["selected_algorithm"],
        "dataset_sha256": dataset_hash,
        "deployment_status": "candidate_only"
    })

    # Log each algorithm's cross-validation result
    for record in result["comparison"]:

        with mlflow.start_run(
            run_name=record["algorithm"],
            nested=True
        ):

            mlflow.log_param(
                "algorithm", record["algorithm"]
            )

            mlflow.log_params(
                record["best_parameters"]
            )

            mlflow.log_metrics({
                "cv_average_precision":
                    record["cv_average_precision"],
                "cv_roc_auc":
                    record["cv_roc_auc"]
            })

            mlflow.set_tag(
                "run_type", "cross_validation_candidate"
            )

    # Held-out metrics belong to selected candidate only
    mlflow.log_metrics({
        "heldout_" + key.lower(): float(value)
        for key, value in result["heldout_test"].items()
        if isinstance(value, (int, float))
        and not isinstance(value, bool)
    })

    mlflow.log_artifact(
        str(output / "candidate_evaluation.json"),
        artifact_path="reports"
    )

    mlflow.log_artifact(
        str(output / "credit_default_candidate.joblib"),
        artifact_path="candidate_model"
    )

    parent_id = parent.info.run_id

print("\nREAL TRAINING AND MLFLOW TRACKING COMPLETE")
print("=" * 50)
print("Selected candidate:", result["selected_algorithm"])
print("Algorithms compared:", len(result["comparison"]))
print("MLflow Run ID:", parent_id)
print("Candidate directory:", output)

print("\nHELD-OUT TEST RESULTS")
print(json.dumps(result["heldout_test"], indent=2))

print("\nDeployed model unchanged.")
