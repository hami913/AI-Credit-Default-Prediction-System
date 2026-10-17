import pytest

from src.risk_scoring import assess_risk


def test_low_risk():
    result = assess_risk(0.20)
    assert result["risk_level"] == "Low"
    assert result["risk_index"] == 20.0


def test_medium_risk():
    result = assess_risk(0.5764544871918588)
    assert result["risk_level"] == "Medium"
    assert result["risk_index"] == 57.65


def test_high_risk():
    result = assess_risk(0.80)
    assert result["risk_level"] == "High"
    assert result["risk_index"] == 80.0


def test_risk_boundaries():
    assert assess_risk(0.30)["risk_level"] == "Medium"
    assert assess_risk(0.60)["risk_level"] == "High"


@pytest.mark.parametrize(
    "probability",
    [-0.1, 1.1, float("nan")]
)
def test_invalid_probabilities(probability):
    with pytest.raises(ValueError):
        assess_risk(probability)


def test_invalid_thresholds():
    with pytest.raises(ValueError):
        assess_risk(
            0.50,
            low_cutoff=0.70,
            high_cutoff=0.40
        )
