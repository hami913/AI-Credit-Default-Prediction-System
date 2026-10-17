import pandas as pd
import pytest

from src.feature_engineering import add_credit_features


def customer(education=2, marriage=1):
    data = {
        "LIMIT_BAL": 200000,
        "SEX": 2,
        "EDUCATION": education,
        "MARRIAGE": marriage,
        "AGE": 30
    }

    for i in [0, 2, 3, 4, 5, 6]:
        data[f"PAY_{i}"] = 0

    for i in range(1, 7):
        data[f"BILL_AMT{i}"] = 10000
        data[f"PAY_AMT{i}"] = 2000

    return pd.DataFrame([data])


@pytest.mark.parametrize("value", [0, 5, 6])
def test_education_grouping(value):
    result = add_credit_features(customer(education=value))
    assert result["EDUCATION"].iloc[0] == 4


def test_marriage_grouping():
    result = add_credit_features(customer(marriage=0))
    assert result["MARRIAGE"].iloc[0] == 3


def test_valid_categories_preserved():
    result = add_credit_features(
        customer(education=2, marriage=1)
    )
    assert result["EDUCATION"].iloc[0] == 2
    assert result["MARRIAGE"].iloc[0] == 1


def test_original_input_unchanged():
    original = customer(education=5, marriage=0)
    before = original.copy(deep=True)

    add_credit_features(original)

    pd.testing.assert_frame_equal(original, before)


def test_unsupported_education_rejected():
    with pytest.raises(ValueError):
        add_credit_features(customer(education=7))


def test_unsupported_marriage_rejected():
    with pytest.raises(ValueError):
        add_credit_features(customer(marriage=9))
