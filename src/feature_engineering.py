
import numpy as np


def add_credit_features(data):
    """
    Convert the 23 original credit features
    into the 38 features expected by the model.
    """

    df_fe = data.copy()

    # Monthly utilization
    utilization_cols = []

    for i in range(1, 7):
        col = f"UTILIZATION_{i}"

        df_fe[col] = (
            df_fe[f"BILL_AMT{i}"]
            / df_fe["LIMIT_BAL"]
        )

        utilization_cols.append(col)

    # Aggregate utilization
    df_fe["AVG_UTILIZATION"] = (
        df_fe[utilization_cols].mean(axis=1)
    )

    df_fe["MAX_UTILIZATION"] = (
        df_fe[utilization_cols].max(axis=1)
    )

    # Repayment behaviour
    pay_status_cols = [
        "PAY_0",
        "PAY_2",
        "PAY_3",
        "PAY_4",
        "PAY_5",
        "PAY_6"
    ]

    df_fe["AVG_DELAY"] = (
        df_fe[pay_status_cols].mean(axis=1)
    )

    df_fe["MAX_DELAY"] = (
        df_fe[pay_status_cols].max(axis=1)
    )

    df_fe["DELAY_MONTHS"] = (
        (df_fe[pay_status_cols] > 0)
        .sum(axis=1)
    )

    # Bill and payment averages
    bill_cols = [
        f"BILL_AMT{i}"
        for i in range(1, 7)
    ]

    payment_cols = [
        f"PAY_AMT{i}"
        for i in range(1, 7)
    ]

    df_fe["AVG_BILL_AMT"] = (
        df_fe[bill_cols].mean(axis=1)
    )

    df_fe["AVG_PAY_AMT"] = (
        df_fe[payment_cols].mean(axis=1)
    )

    # Payment ratios
    df_fe["PAY_TO_BILL_RATIO"] = np.where(
        df_fe["AVG_BILL_AMT"] > 0,
        df_fe["AVG_PAY_AMT"]
        / df_fe["AVG_BILL_AMT"],
        0
    )

    df_fe["PAY_TO_LIMIT_RATIO"] = (
        df_fe["AVG_PAY_AMT"]
        / df_fe["LIMIT_BAL"]
    )

    return df_fe
