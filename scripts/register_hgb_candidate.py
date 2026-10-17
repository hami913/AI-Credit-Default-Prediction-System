import hashlib
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

from src.train import validate_training_data
from src.feature_engineering import add_credit_features


ROOT = Path(__file__).resolve().parents[1]
URI = "http://127.0.0.1:5000"

MODEL_NAME = "CreditRisk_HGB_38Feature_Candidate"

DATA = Path(
    r"C:\Users\mianh\Downloads\UCI_Credit_Card.csv"
)

SOURCE = (
    ROOT / "models" / "candidates"
    / "real_20261017_145037_cfaac2"
    / "credit_default_candidate.joblib"
)

if not DATA.is_file() or not SOURCE.is_file():
    raise FileNotFoundError("Dataset or candidate model missing.")

with SOURCE.open("rb") as f:
    source_hash = hashlib.file_digest(
        f, "sha256"
    ).hexdigest()

mlflow.set_tracking_uri(URI)
mlflow.set_registry_uri(URI)

client = MlflowClient(tracking_uri=URI)

# Do not register the same candidate twice.
existing = client.search_model_versions(
    f"name='{MODEL_NAME}'"
)

for version in existing:
    if version.tags.get("source_sha256") == source_hash:
        print("\nMODEL ALREADY REGISTERED")
        print("Name:", MODEL_NAME)
        print("Version:", version.version)
        print("Approval: not approved")
        raise SystemExit(0)

artifact = joblib.load(SOURCE)

if artifact["model_name"] != "hist_gradient_boosting":
    raise ValueError("Unexpected candidate model.")

features = list(artifact["features"])
pipeline = artifact["pipeline"]

if len(features) != 38:
    raise ValueError("Expected exactly 38 model features.")

# Prepare a small real-data input example.
# This does NOT retrain or evaluate the model.
data = pd.read_csv(DATA)
clean, raw_features = validate_training_data(data)

example = add_credit_features(
    clean[raw_features].head(5).copy()
).loc[:, features]

probabilities = pipeline.predict_proba(example)

if probabilities.shape != (5, 2):
    raise RuntimeError("Unexpected probability output shape.")

signature = infer_signature(
    example,
    probabilities
)

fairness_reports = sorted(
    (ROOT / "reports").glob("real_fairness_audit_*.json"),
    key=lambda p: p.stat().st_mtime
)

if not fairness_reports:
    raise FileNotFoundError("Fairness audit report missing.")

mlflow.set_experiment(
    "Credit_Risk_Model_Registration"
)

with mlflow.start_run(
    run_name="HGB_Candidate_Registration"
) as run:

    mlflow.log_params({
        "algorithm": "hist_gradient_boosting",
        "feature_count": len(features),
        "candidate_status": "not_approved",
        "input_type": "engineered_features",
        "probability_columns": "class_0,class_1"
    })

    mlflow.set_tags({
        "run_type": "candidate_registration",
        "source_sha256": source_hash,
        "deployment_status": "not_approved",
        "fairness_review": "diagnostic_completed",
        "regulatory_compliance": "not_assessed"
    })

    mlflow.log_artifact(
        str(fairness_reports[-1]),
        artifact_path="fairness_audit"
    )

    model_info = mlflow.sklearn.log_model(
        sk_model=pipeline,
        name="probability_pipeline",
        signature=signature,
        input_example=example,
        pyfunc_predict_fn="predict_proba",
        serialization_format="cloudpickle",
        registered_model_name=MODEL_NAME,
        metadata={
            "feature_count": 38,
            "feature_engineering_required": True,
            "feature_engineering_module":
                "src.feature_engineering",
            "probability_output":
                "column_0_nondefault_column_1_default",
            "deployment_approved": False
        }
    )

    run_id = run.info.run_id

version_number = model_info.registered_model_version

if version_number is None:
    raise RuntimeError("Registry did not return a model version.")

# Explicitly mark the registered version as not approved.
client.set_model_version_tag(
    MODEL_NAME,
    version_number,
    "approval_status",
    "not_approved"
)

client.set_model_version_tag(
    MODEL_NAME,
    version_number,
    "source_sha256",
    source_hash
)

client.set_model_version_tag(
    MODEL_NAME,
    version_number,
    "expected_features",
    "38_engineered"
)

client.set_model_version_tag(
    MODEL_NAME,
    version_number,
    "deployment_status",
    "candidate_only"
)

# Check registered model gives the same probabilities.
registered = mlflow.sklearn.load_model(
    f"models:/{MODEL_NAME}/{version_number}"
)

np.testing.assert_allclose(
    registered.predict_proba(example),
    probabilities,
    rtol=1e-10,
    atol=1e-12
)

with SOURCE.open("rb") as f:
    hash_after = hashlib.file_digest(
        f, "sha256"
    ).hexdigest()

if source_hash != hash_after:
    raise RuntimeError("Candidate artifact changed unexpectedly.")

print("\nMLFLOW MODEL REGISTRATION SUCCESSFUL")
print("=" * 60)
print("Registered model:", MODEL_NAME)
print("Model version:", version_number)
print("Run ID:", run_id)
print("Expected features:", len(features))
print("Probability predictions: verified")
print("Approval status: NOT APPROVED")
print("Production deployment: NOT PERFORMED")
print("Original model: UNCHANGED")
print("Candidate artifact: UNCHANGED")
