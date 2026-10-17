
import pytest
from src.evaluate import evaluate_model


def test_perfect_predictions():
    result = evaluate_model(
        [0, 0, 1, 1],
        [0.1, 0.2, 0.8, 0.9]
    )

    assert result["Accuracy"] == 1.0
    assert result["F1"] == 1.0
    assert result["ROC_AUC"] == 1.0
    assert result["PR_AUC"] == 1.0


def test_invalid_probabilities():
    with pytest.raises(ValueError):
        evaluate_model([0, 1], [0.2, 1.5])


def test_mismatched_lengths():
    with pytest.raises(ValueError):
        evaluate_model([0, 1, 1], [0.2, 0.8])


def test_single_class():
    result = evaluate_model(
        [0, 0, 0],
        [0.1, 0.2, 0.3]
    )

    assert result["ROC_AUC"] is None
    assert result["PR_AUC"] is None
