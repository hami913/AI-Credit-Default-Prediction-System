"""Credit risk probability drift and performance monitoring."""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.evaluate import evaluate_model


def _validate_probabilities(values, name):
    arr = np.asarray(values, dtype=float)

    if arr.ndim != 1 or len(arr) == 0:
        raise ValueError(f"{name} must be a nonempty 1D array.")

    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains nonfinite values.")

    if np.any((arr < 0) | (arr > 1)):
        raise ValueError(f"{name} must contain probabilities 0 to 1.")

    return arr


def probability_drift(
    reference_probabilities,
    current_probabilities,
    bins=10,
    min_samples=30
):
    """
    Compare predicted default probability distributions.

    PSI thresholds are illustrative monitoring heuristics.
    They must not be treated as regulatory rules.
    """

    reference = _validate_probabilities(
        reference_probabilities, "reference"
    )

    current = _validate_probabilities(
        current_probabilities, "current"
    )

    if isinstance(bins, bool) or not isinstance(bins, int) or bins < 2:
        raise ValueError("bins must be an integer >= 2.")

    if (
        isinstance(min_samples, bool)
        or not isinstance(min_samples, int)
        or min_samples < 1
    ):
        raise ValueError("min_samples must be a positive integer.")

    edges = np.linspace(0, 1, bins + 1)

    reference_counts, _ = np.histogram(
        reference, bins=edges
    )

    current_counts, _ = np.histogram(
        current, bins=edges
    )

    # Smoothing prevents log(0)
    smoothing = 0.5

    ref_pct = (
        reference_counts + smoothing
    ) / (len(reference) + smoothing * bins)

    cur_pct = (
        current_counts + smoothing
    ) / (len(current) + smoothing * bins)

    psi = float(
        np.sum(
            (cur_pct - ref_pct) *
            np.log(cur_pct / ref_pct)
        )
    )

    enough_data = (
        len(reference) >= min_samples
        and len(current) >= min_samples
    )

    if not enough_data:
        status = "insufficient_samples"
    elif psi >= 0.25:
        status = "high"
    elif psi >= 0.10:
        status = "moderate"
    else:
        status = "low"

    return {
        "reference_count": int(len(reference)),
        "current_count": int(len(current)),
        "reference_mean_probability": float(reference.mean()),
        "current_mean_probability": float(current.mean()),
        "mean_probability_change": float(
            current.mean() - reference.mean()
        ),
        "psi": psi,
        "drift_status": status,
        "minimum_samples": min_samples
    }


def build_monitoring_report(
    reference_probabilities,
    current_probabilities,
    actual_labels=None,
    threshold=0.50,
    model_version="unverified"
):
    """
    Create drift report and optional performance evaluation.

    actual_labels must correspond to current predictions.
    """

    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Invalid prediction threshold.")

    drift = probability_drift(
        reference_probabilities,
        current_probabilities
    )

    performance = None

    if actual_labels is not None:
        performance = evaluate_model(
            actual_labels,
            current_probabilities,
            threshold=threshold
        )

    return {
        "report_type": "credit_risk_model_monitoring",
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "model_version": model_version,
        "threshold": float(threshold),
        "probability_drift": drift,
        "performance": performance,
        "actual_labels_available": actual_labels is not None,
        "limitations": [
            "PSI indicates distribution change, not the cause.",
            "Probability drift alone does not prove model degradation.",
            "Performance metrics require real observed outcomes.",
            "PSI alert thresholds are illustrative, not regulatory."
        ]
    }


def save_monitoring_report(report, path, overwrite=False):
    output = Path(path)

    if output.exists() and not overwrite:
        raise FileExistsError(
            f"Monitoring report already exists: {output}"
        )

    output.parent.mkdir(
        parents=True, exist_ok=True
    )

    with output.open("w", encoding="utf-8") as f:
        json.dump(
            report, f, indent=4, allow_nan=False
        )

    return output


if __name__ == "__main__":
    # Synthetic distributions for technical demonstration
    rng = np.random.default_rng(42)

    reference = rng.beta(2.0, 6.0, 250)
    current = rng.beta(2.4, 5.4, 250)

    report = build_monitoring_report(
        reference,
        current,
        model_version="synthetic-demo"
    )

    path = Path("reports/demo_model_monitoring.json")

    if not path.exists():
        save_monitoring_report(report, path)
        print("Demo report saved:", path)
    else:
        print("Existing demo report preserved:", path)

    print("MODEL MONITORING")
    print("Reference records:", len(reference))
    print("Current records:", len(current))
    print("PSI:", round(
        report["probability_drift"]["psi"], 4
    ))
    print("Drift status:",
          report["probability_drift"]["drift_status"])
    print("Real performance available:",
          report["actual_labels_available"])
