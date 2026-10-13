import numpy as np
import pandas as pd
import pytest

from src.feature_engineering import add_credit_features


@pytest.fixture
def customer():
    data = {
        "LIMIT_BAL": 200000,
        "SEX": 2,
        "EDUCATION": 2,
        "MARRIAGE": 1,
        "AGE": 30,
        "PAY_0": 0,
        "PAY_2": 1,
        "PAY_3": 0,
        "PAY_4": 2,
        "PAY_5": 0,
        "PAY_6": 0,
    }

    for i in range(1, 7):
        data[f"BILL_AMT{i}"] = 20000 * i
        data[f"PAY_AMT{i}"] = 5000 * i

    return pd.DataFrame([data])


def test_total_features(customer):
    result = add_credit_features(customer)
    assert result.shape[1] == 38


def test_monthly_utilization(customer):
    result = add_credit_features(customer)

    for i in range(1, 7):
        expected = (20000 * i) / 200000
        assert result[f"UTILIZATION_{i}"].iloc[0] == pytest.approx(expected)


def test_average_utilization(customer):
    result = add_credit_features(customer)
    assert result["AVG_UTILIZATION"].iloc[0] == pytest.approx(0.35)


def test_max_utilization(customer):
    result = add_credit_features(customer)
    assert result["MAX_UTILIZATION"].iloc[0] == pytest.approx(0.60)


def test_average_delay(customer):
    result = add_credit_features(customer)
    assert result["AVG_DELAY"].iloc[0] == pytest.approx(0.5)


def test_max_delay(customer):
    result = add_credit_features(customer)
    assert result["MAX_DELAY"].iloc[0] == 2


def test_delayed_months(customer):
    result = add_credit_features(customer)
    assert result["DELAY_MONTHS"].iloc[0] == 2


def test_average_bill(customer):
    result = add_credit_features(customer)
    assert result["AVG_BILL_AMT"].iloc[0] == pytest.approx(70000)


def test_average_payment(customer):
    result = add_credit_features(customer)
    assert result["AVG_PAY_AMT"].iloc[0] == pytest.approx(17500)


def test_payment_to_bill_ratio(customer):
    result = add_credit_features(customer)
    assert result["PAY_TO_BILL_RATIO"].iloc[0] == pytest.approx(0.25)


def test_payment_to_limit_ratio(customer):
    result = add_credit_features(customer)
    assert result["PAY_TO_LIMIT_RATIO"].iloc[0] == pytest.approx(0.0875)


def test_zero_bill_amount(customer):
    for i in range(1, 7):
        customer[f"BILL_AMT{i}"] = 0

    result = add_credit_features(customer)

    assert result["PAY_TO_BILL_RATIO"].iloc[0] == 0


def test_original_data_unchanged(customer):
    original = customer.copy(deep=True)

    add_credit_features(customer)

    pd.testing.assert_frame_equal(customer, original)


def test_batch_feature_engineering(customer):
    batch = pd.concat([customer] * 5, ignore_index=True)

    result = add_credit_features(batch)

    assert result.shape == (5, 38)
    assert np.isfinite(result.to_numpy(dtype=float)).all()
