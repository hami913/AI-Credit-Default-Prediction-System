import json

import numpy as np
import pytest

from src.monitoring import (
    probability_drift,
    build_monitoring_report,
    save_monitoring_report
)


def test_identical_distributions():
    values = np.tile([0.1, 0.3, 0.7, 0.9], 20)

    result = probability_drift(values, values)

    assert result["psi"] == pytest.approx(0.0)
    assert result["drift_status"] == "low"


def test_large_distribution_shift():
    reference = np.full(80, 0.2)
    current = np.full(80, 0.8)

    result = probability_drift(reference, current)

    assert result["psi"] > 0.25
    assert result["drift_status"] == "high"


def test_invalid_probability():
    with pytest.raises(ValueError):
        probability_drift(
            [0.1, 0.2],
            [-0.1, 0.5]
        )


def test_nan_probability():
    with pytest.raises(ValueError):
        probability_drift(
            [0.1, np.nan],
            [0.2, 0.3]
        )


def test_insufficient_samples():
    result = probability_drift(
        [0.1, 0.2],
        [0.4, 0.5]
    )

    assert result["drift_status"] == "insufficient_samples"


def test_performance_with_labels():
    labels = np.array([0, 1] * 40)
    probabilities = np.where(labels == 1, 0.9, 0.1)

    report = build_monitoring_report(
        probabilities,
        probabilities,
        actual_labels=labels
    )

    assert report["actual_labels_available"] is True
    assert report["performance"]["Accuracy"] == 1.0


def test_performance_without_labels():
    values = np.tile([0.1, 0.7], 40)

    report = build_monitoring_report(
        values, values
    )

    assert report["performance"] is None
    assert report["actual_labels_available"] is False


def test_save_and_protect_report(tmp_path):
    values = np.tile([0.1, 0.8], 40)

    report = build_monitoring_report(
        values, values
    )

    path = tmp_path / "monitoring.json"

    save_monitoring_report(report, path)

    saved = json.loads(
        path.read_text(encoding="utf-8")
    )

    assert saved["report_type"] == "credit_risk_model_monitoring"

    with pytest.raises(FileExistsError):
        save_monitoring_report(report, path)
