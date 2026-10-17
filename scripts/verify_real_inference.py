import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.preprocessing import prepare_model_input

ROOT = Path.cwd()

DATA_PATH = Path(
    r"C:\Users\mianh\Downloads\UCI_Credit_Card.csv"
)

MODEL_PATH = ROOT / "models" / "credit_default_model_v1.joblib"

data = pd.read_csv(DATA_PATH)
artifact = joblib.load(MODEL_PATH)

with open(
    ROOT / "config" / "input_features.json",
    encoding="utf-8"
) as f:
    raw_features = json.load(f)

print("\nREAL DATA INFERENCE VERIFICATION")
print("=" * 45)

print("Dataset rows:", len(data))
print("Model features:", len(artifact["features"]))

# Prepare all customer records
X = prepare_model_input(
    data[raw_features],
    artifact["features"]
)

# Predict default probabilities
probabilities = artifact["pipeline"].predict_proba(X)[:, 1]

threshold = float(artifact.get("threshold", 0.5))

predictions = (probabilities >= threshold).astype(int)

# Validate predictions
assert len(probabilities) == len(data)
assert np.isfinite(probabilities).all()
assert ((probabilities >= 0) & (probabilities <= 1)).all()

report = {
    "verification_type": "inference_only",
    "model_version": str(artifact["model_version"]),
    "total_records": int(len(data)),
    "input_features": int(X.shape[1]),
    "threshold": threshold,
    "average_default_probability": float(probabilities.mean()),
    "minimum_probability": float(probabilities.min()),
    "maximum_probability": float(probabilities.max()),
    "predicted_default_count": int(predictions.sum()),
    "predicted_nondefault_count": int((predictions == 0).sum()),
    "independent_test_performance": "not_evaluated"
}

for key, value in report.items():
    print(f"{key}: {value}")

output_dir = ROOT / "reports"
output_dir.mkdir(exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

output = output_dir / f"real_data_inference_{timestamp}.json"

with open(output, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=4)

print("\nVerification completed.")
print("Report:", output)
print("Saved model unchanged.")
