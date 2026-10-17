import pandas as pd
import pytest

from src.fairness import (
    fairness_by_group,
    fairness_gaps
)


@pytest.fixture
def sample_data():
    return pd.DataFrame({
        "SEX": [1, 1, 1, 1, 2, 2, 2, 2],
        "actual_default": [0, 0, 1, 1, 0, 0, 1, 1],
        "default_probability": [
            0.1, 0.7, 0.8, 0.3,
            0.2, 0.4, 0.6, 0.9
        ]
    })


def test_two_groups(sample_data):
    result = fairness_by_group(
        sample_data, min_group_size=3
    )
    assert len(result) == 2


def test_confusion_metrics(sample_data):
    result = fairness_by_group(
        sample_data, min_group_size=3
    )
    group_one = result.set_index("group").loc["1"]

    assert group_one["TP"] == 1
    assert group_one["FP"] == 1
    assert group_one["TPR"] == pytest.approx(0.5)
    assert group_one["FPR"] == pytest.approx(0.5)


def test_fairness_gaps(sample_data):
    result = fairness_by_group(
        sample_data, min_group_size=3
    )
    gaps = fairness_gaps(result)

    assert gaps["TPR_gap"] == pytest.approx(0.5)
    assert gaps["FPR_gap"] == pytest.approx(0.5)


def test_invalid_probability(sample_data):
    sample_data.loc[0, "default_probability"] = 1.5

    with pytest.raises(ValueError):
        fairness_by_group(sample_data)


def test_invalid_actual_label(sample_data):
    sample_data.loc[0, "actual_default"] = 3

    with pytest.raises(ValueError):
        fairness_by_group(sample_data)


def test_unknown_group(sample_data):
    sample_data.loc[0, "SEX"] = None

    result = fairness_by_group(
        sample_data, min_group_size=1
    )

    assert "Unknown" in result["group"].tolist()


def test_small_group_exclusion(sample_data):
    result = fairness_by_group(
        sample_data, min_group_size=10
    )

    gaps = fairness_gaps(result)

    assert gaps["FPR_gap"] is None


def test_missing_column(sample_data):
    with pytest.raises(ValueError):
        fairness_by_group(
            sample_data.drop(columns=["SEX"])
        )
