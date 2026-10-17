import json

import pandas as pd
import pytest

from src.reporting import (
    build_audit_report,
    save_audit_report
)


@pytest.fixture
def audit_data():
    return pd.DataFrame({
        "SEX": [1, 1, 1, 1, 2, 2, 2, 2],
        "actual_default": [0, 0, 1, 1, 0, 0, 1, 1],
        "default_probability": [
            0.1, 0.7, 0.8, 0.3,
            0.2, 0.4, 0.6, 0.9
        ]
    })


def test_report_creation(audit_data):
    report = build_audit_report(
        audit_data, min_group_size=3
    )
    assert report["total_customers"] == 8


def test_evaluation_metrics(audit_data):
    report = build_audit_report(
        audit_data, min_group_size=3
    )

    assert "ROC_AUC" in report["model_evaluation"]
    assert "Brier_Score" in report["model_evaluation"]


def test_fairness_in_report(audit_data):
    report = build_audit_report(
        audit_data, min_group_size=3
    )

    fairness = report["fairness_analysis"]

    assert len(fairness["groups"]) == 2
    assert fairness["disparities"]["TPR_gap"] == pytest.approx(0.5)


def test_report_json_serializable(audit_data):
    report = build_audit_report(
        audit_data, min_group_size=3
    )

    json.dumps(report, allow_nan=False)


def test_report_file_saved(audit_data, tmp_path):
    report = build_audit_report(
        audit_data, min_group_size=3
    )

    path = tmp_path / "audit.json"

    save_audit_report(report, path)

    assert path.exists()

    saved = json.loads(path.read_text(encoding="utf-8"))

    assert saved["total_customers"] == 8


def test_existing_report_protected(audit_data, tmp_path):
    report = build_audit_report(
        audit_data, min_group_size=3
    )

    path = tmp_path / "audit.json"

    save_audit_report(report, path)

    with pytest.raises(FileExistsError):
        save_audit_report(report, path)


def test_missing_labels_rejected(audit_data):
    data = audit_data.drop(columns=["actual_default"])

    with pytest.raises(ValueError):
        build_audit_report(data)


def test_compliance_not_assumed(audit_data):
    report = build_audit_report(
        audit_data, min_group_size=3
    )

    assert report["regulatory_compliance"] == "not_assessed"
