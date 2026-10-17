import json
import hashlib
from pathlib import Path
import mlflow

ROOT = Path(__file__).resolve().parent.parent

reports = list(
    (ROOT / "reports").glob("real_data_inference_*.json")
)

if not reports:
    raise FileNotFoundError("No inference report found.")

report_path = max(reports, key=lambda p: p.stat().st_mtime)

with report_path.open(encoding="utf-8") as f:
    report = json.load(f)

model_path = ROOT / "models" / "credit_default_model_v1.joblib"

model_hash = hashlib.sha256(
    model_path.read_bytes()
).hexdigest()

mlflow.set_tracking_uri("http://127.0.0.1:5000")

mlflow.set_experiment(
    "Credit_Risk_Inference_Verification"
)

with mlflow.start_run(
    run_name="Random_Forest_v1_Inference"
) as run:

    mlflow.log_params({
        "model_version": str(report["model_version"]),
        "feature_count": int(report["input_features"]),
        "threshold": float(report["threshold"])
    })

    mlflow.log_metrics({
        "total_records": report["total_records"],
        "mean_default_probability":
            report["average_default_probability"],
        "predicted_defaults":
            report["predicted_default_count"],
        "predicted_nondefaults":
            report["predicted_nondefault_count"]
    })

    mlflow.set_tags({
        "model_type": "Random Forest",
        "model_sha256": model_hash,
        "run_type": "inference_verification",
        "evaluation_status": "not_independently_evaluated"
    })

    mlflow.log_artifact(str(report_path))

    print("\nMLFLOW EXPERIMENT SUCCESSFULLY LOGGED")
    print("Experiment: Credit_Risk_Inference_Verification")
    print("Run ID:", run.info.run_id)
    print("Records:", report["total_records"])
    print("Feature Count:", report["input_features"])
    print("Model unchanged.")
