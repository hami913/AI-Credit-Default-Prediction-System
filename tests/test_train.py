from pathlib import Path
import json
import pytest
import joblib
from src.train import (
    TARGET, validate_training_data, synthetic_data, train_candidate
)


def test_synthetic_schema():
    data = synthetic_data()
    clean, raw = validate_training_data(data)
    assert len(raw) == 23
    assert len(clean) == 120
    assert TARGET in clean.columns


def test_missing_target_rejected():
    with pytest.raises(ValueError, match="Missing training columns"):
        validate_training_data(synthetic_data().drop(columns=[TARGET]))


def test_three_algorithms_cv_and_candidate(tmp_path):
    report = train_candidate(
        synthetic_data(), tmp_path, folds=2, iterations=1
    )
    assert len(report["comparison"]) == 3
    assert report["feature_count"] == 38
    assert report["heldout_test_rows"] == 24
    artifact = joblib.load(tmp_path / "credit_default_candidate.joblib")
    assert len(artifact["features"]) == 38
    assert set(artifact["pipeline"].named_steps) == {"preprocessor", "classifier"}
    saved_report = json.loads((tmp_path / "candidate_evaluation.json").read_text())
    assert saved_report["selected_algorithm"] == report["selected_algorithm"]


def test_existing_candidate_protected(tmp_path):
    (tmp_path / "credit_default_candidate.joblib").write_bytes(b"keep-original")
    with pytest.raises(FileExistsError):
        train_candidate(synthetic_data(), tmp_path, folds=2, iterations=1)
    assert (tmp_path / "credit_default_candidate.joblib").read_bytes() == b"keep-original"
