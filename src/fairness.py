import numpy as np
import pandas as pd


def _safe_divide(numerator, denominator):
    if denominator == 0:
        return None
    return float(numerator / denominator)


def fairness_by_group(
    data,
    group_column="SEX",
    label_column="actual_default",
    probability_column="default_probability",
    threshold=0.50,
    min_group_size=30
):
    """
    Evaluate model predictions across demographic groups.

    Requires actual outcomes and predicted probabilities.
    Intended for model auditing, not automated lending decisions.
    """

    if not isinstance(data, pd.DataFrame) or data.empty:
        raise ValueError("Provide a nonempty DataFrame.")

    required = [
        group_column, label_column, probability_column
    ]

    if any(col not in data.columns for col in required):
        raise ValueError("Required audit columns are missing.")

    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Invalid threshold.")

    if min_group_size < 1:
        raise ValueError("min_group_size must be positive.")

    df = data.copy()

    y = pd.to_numeric(df[label_column], errors="raise")
    probabilities = pd.to_numeric(
        df[probability_column], errors="raise"
    )

    if not y.isin([0, 1]).all():
        raise ValueError("Actual labels must be 0 or 1.")

    if not np.isfinite(probabilities.to_numpy()).all():
        raise ValueError("Probabilities must be finite.")

    if not probabilities.between(0, 1).all():
        raise ValueError("Probabilities must be between 0 and 1.")

    df["_actual"] = y.to_numpy(dtype=int)
    df["_probability"] = probabilities.to_numpy()
    df["_predicted"] = (
        probabilities.to_numpy() >= threshold
    ).astype(int)

    df["_group"] = (
        df[group_column].astype("string").fillna("Unknown")
    )

    results = []

    for group, sub in df.groupby("_group"):
        actual = sub["_actual"].to_numpy()
        predicted = sub["_predicted"].to_numpy()
        probability = sub["_probability"].to_numpy()

        tp = int(((actual == 1) & (predicted == 1)).sum())
        tn = int(((actual == 0) & (predicted == 0)).sum())
        fp = int(((actual == 0) & (predicted == 1)).sum())
        fn = int(((actual == 1) & (predicted == 0)).sum())

        n = len(sub)

        results.append({
            "group": str(group),
            "sample_size": n,
            "actual_defaults": int(actual.sum()),
            "prediction_rate": float(predicted.mean()),
            "observed_default_rate": float(actual.mean()),
            "mean_predicted_probability": float(
                probability.mean()
            ),
            "TPR": _safe_divide(tp, tp + fn),
            "FPR": _safe_divide(fp, fp + tn),
            "FNR": _safe_divide(fn, fn + tp),
            "precision": _safe_divide(tp, tp + fp),
            "brier_score": float(
                np.mean((probability - actual) ** 2)
            ),
            "TP": tp,
            "TN": tn,
            "FP": fp,
            "FN": fn,
            "sufficient_sample": n >= min_group_size
        })

    return pd.DataFrame(results)


def fairness_gaps(group_results):
    """
    Calculate max-minus-min differences across groups.

    Small groups and undefined rates are excluded.
    """

    eligible = group_results[
        group_results["sufficient_sample"]
    ]

    metrics = [
        "prediction_rate",
        "TPR",
        "FPR",
        "FNR",
        "observed_default_rate",
        "mean_predicted_probability"
    ]

    gaps = {}

    for metric in metrics:
        values = eligible[metric].dropna()

        if len(values) < 2:
            gaps[metric + "_gap"] = None
        else:
            gaps[metric + "_gap"] = float(
                values.max() - values.min()
            )

    return gaps


if __name__ == "__main__":
    # Synthetic demonstration data, not real customer data
    sample = pd.DataFrame({
        "SEX": [1, 1, 1, 1, 2, 2, 2, 2],
        "actual_default": [0, 0, 1, 1, 0, 0, 1, 1],
        "default_probability": [
            0.1, 0.7, 0.8, 0.3,
            0.2, 0.4, 0.6, 0.9
        ]
    })

    report = fairness_by_group(
        sample,
        min_group_size=3
    )

    print("FAIRNESS AUDIT")
    print(report.to_string(index=False))

    print("\nGROUP DISPARITIES")
    print(fairness_gaps(report))
