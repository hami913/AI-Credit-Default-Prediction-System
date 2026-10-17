import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluate import evaluate_model
from src.fairness import fairness_by_group, fairness_gaps


def _native(value):
    """Convert NumPy values into JSON-safe Python values."""

    if isinstance(value, dict):
        return {k: _native(v) for k, v in value.items()}

    if isinstance(value, list):
        return [_native(v) for v in value]

    if isinstance(value, np.generic):
        return value.item()

    return value


def build_audit_report(
    data,
    model_version="unverified",
    group_column="SEX",
    threshold=0.50,
    min_group_size=30,
    data_source="unspecified"
):
    """
    Build a model validation and fairness audit report.

    Required columns:
    - actual_default
    - default_probability
    - demographic group such as SEX

    This is an audit-style report, not proof of
    regulatory compliance.
    """

    if not isinstance(data, pd.DataFrame):
        raise TypeError("Data must be a DataFrame.")

    required = {
        "actual_default",
        "default_probability",
        group_column
    }

    missing = required - set(data.columns)

    if missing:
        raise ValueError(
            f"Missing report columns: {sorted(missing)}"
        )

    metrics = evaluate_model(
        data["actual_default"],
        data["default_probability"],
        threshold=threshold
    )

    group_metrics = fairness_by_group(
        data,
        group_column=group_column,
        threshold=threshold,
        min_group_size=min_group_size
    )

    gaps = fairness_gaps(group_metrics)

    report = {
        "report_type": "credit_risk_model_audit",
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "model_version": model_version,
        "data_source": data_source,
        "total_customers": int(len(data)),
        "threshold": float(threshold),
        "model_evaluation": _native(metrics),
        "fairness_analysis": {
            "group_column": group_column,
            "min_group_size": min_group_size,
            "groups": json.loads(
                group_metrics.to_json(orient="records")
            ),
            "disparities": _native(gaps)
        },
        "regulatory_compliance": "not_assessed",
        "limitations": [
            "Group differences do not independently prove discrimination.",
            "Small groups may produce unreliable estimates.",
            "Formal regulatory compliance has not been assessed."
        ]
    }

    return report


def save_audit_report(
    report,
    output_path,
    overwrite=False
):
    """Save an audit report as JSON."""

    path = Path(output_path)

    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Report already exists: {path}"
        )

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            _native(report),
            f,
            indent=4,
            allow_nan=False
        )

    return path


if __name__ == "__main__":

    # Synthetic data for functionality testing only
    sample = pd.DataFrame({
        "SEX": [1, 1, 1, 1, 2, 2, 2, 2],
        "actual_default": [0, 0, 1, 1, 0, 0, 1, 1],
        "default_probability": [
            0.1, 0.7, 0.8, 0.3,
            0.2, 0.4, 0.6, 0.9
        ]
    })

    report = build_audit_report(
        sample,
        model_version="synthetic-demo",
        data_source="synthetic",
        min_group_size=3
    )

    output = save_audit_report(
        report,
        "reports/demo_credit_risk_audit.json",
        overwrite=True
    )

    print("AUDIT REPORT GENERATED")
    print("File:", output)
    print("Customers:", report["total_customers"])
    print("Accuracy:",
          report["model_evaluation"]["Accuracy"])
    print("Compliance:",
          report["regulatory_compliance"])
