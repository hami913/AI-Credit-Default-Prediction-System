import json
from pathlib import Path
from datetime import datetime, timezone

import mlflow


ROOT = Path(__file__).resolve().parents[1]

reports = sorted(
    (ROOT / "reports").glob(
        "candidate_threshold_validation_*.json"
    ),
    key=lambda p: p.stat().st_mtime
)

if not reports:
    raise FileNotFoundError("Threshold validation report missing.")

source = reports[-1]

with source.open(encoding="utf-8") as f:
    report = json.load(f)

baseline = report["baseline"]
proposed = report["proposed_threshold"]

# Hypothetical relative costs (not real bank costs)
cost_ratios = [1, 2, 3, 5, 10]

results = []

print("\nBUSINESS COST SENSITIVITY ANALYSIS")
print("=" * 70)

print(
    f"{'FN/FP Cost Ratio':<20}"
    f"{'Threshold 0.50':>20}"
    f"{'Threshold 0.25':>20}"
)

for ratio in cost_ratios:

    # Assign FP cost = 1 unit, FN cost = ratio units
    old_cost = (
        baseline["FP"] +
        ratio * baseline["FN"]
    )

    new_cost = (
        proposed["FP"] +
        ratio * proposed["FN"]
    )

    if old_cost < new_cost:
        lower_cost = 0.50
    elif new_cost < old_cost:
        lower_cost = 0.25
    else:
        lower_cost = None

    results.append({
        "false_positive_cost_units": 1,
        "false_negative_cost_units": ratio,
        "cost_at_050": old_cost,
        "cost_at_025": new_cost,
        "lower_cost_threshold": lower_cost
    })

    print(
        f"{ratio:<20}"
        f"{old_cost:>20,.2f}"
        f"{new_cost:>20,.2f}"
    )

avoided_fn = baseline["FN"] - proposed["FN"]
additional_fp = proposed["FP"] - baseline["FP"]

break_even_ratio = (
    additional_fp / avoided_fn
    if avoided_fn > 0
    else None
)

print("\nADDITIONAL DEFAULTS DETECTED:", avoided_fn)
print("ADDITIONAL FALSE POSITIVES:", additional_fp)

print(
    "Break-even FN/FP cost ratio:",
    round(break_even_ratio, 4)
    if break_even_ratio is not None
    else "Not applicable"
)

output_report = {
    "analysis_type": "hypothetical_cost_sensitivity",
    "source_report": source.name,
    "heldout_records": report["heldout_records"],
    "model": report["model"],
    "cost_scenarios": results,
    "break_even_fn_fp_cost_ratio": break_even_ratio,
    "threshold_selection": "not_performed",
    "deployment_approved": False,
    "limitations": [
        "Costs are illustrative, not measured bank losses.",
        "This analysis does not authorize a threshold change.",
        "Held-out results must not be used for repeated tuning.",
        "Calibration and fairness require further assessment."
    ]
}

stamp = datetime.now(timezone.utc).strftime(
    "%Y%m%d_%H%M%S"
)

output = ROOT / "reports" / (
    f"business_cost_analysis_{stamp}.json"
)

output.write_text(
    json.dumps(output_report, indent=2),
    encoding="utf-8"
)

# Log business cost analysis
mlflow.set_tracking_uri("http://127.0.0.1:5000")

mlflow.set_experiment(
    "Credit_Risk_Business_Cost_Audit"
)

with mlflow.start_run(
    run_name="HGB_Fixed_Threshold_Cost_Sensitivity"
):
    mlflow.log_params({
        "model": report["model"],
        "baseline_threshold": 0.50,
        "proposed_threshold": 0.25,
        "fp_cost_unit": 1
    })

    mlflow.log_metrics({
        "additional_defaults_detected": avoided_fn,
        "additional_false_positives": additional_fp,
        "break_even_cost_ratio": break_even_ratio
    })

    mlflow.set_tags({
        "analysis_type": "hypothetical_cost_sensitivity",
        "deployment_status": "not_approved"
    })

    mlflow.log_artifact(str(output))

print("\nReport saved:", output)
print("MLflow logging complete.")
print("Existing deployed model unchanged.")
