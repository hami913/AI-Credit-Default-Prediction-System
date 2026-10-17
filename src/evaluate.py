
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    brier_score_loss,
)


def evaluate_model(y_true, y_prob, threshold=0.50):
    """
    Evaluate binary credit default predictions.

    y_true: Actual labels (0 = No Default, 1 = Default)
    y_prob: Predicted probability of default
    threshold: Classification threshold
    """

    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob, dtype=float).ravel()

    if len(y_true) == 0:
        raise ValueError("Evaluation data cannot be empty.")

    if len(y_true) != len(y_prob):
        raise ValueError("Labels and probabilities must have equal length.")

    if not np.isin(y_true, [0, 1]).all():
        raise ValueError("Labels must be binary: 0 or 1.")

    if not np.isfinite(y_prob).all():
        raise ValueError("Probabilities must be finite.")

    if np.any((y_prob < 0) | (y_prob > 1)):
        raise ValueError("Probabilities must be between 0 and 1.")

    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must be between 0 and 1.")

    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    results = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(
            y_true, y_pred, zero_division=0
        ),
        "Recall": recall_score(
            y_true, y_pred, zero_division=0
        ),
        "F1": f1_score(
            y_true, y_pred, zero_division=0
        ),
        "Specificity": (
            tn / (tn + fp) if (tn + fp) else 0.0
        ),
        "Brier_Score": brier_score_loss(y_true, y_prob),
        "Threshold": threshold,
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }

    # AUC metrics need both classes
    if len(np.unique(y_true)) == 2:
        results["ROC_AUC"] = roc_auc_score(
            y_true, y_prob
        )
        results["PR_AUC"] = average_precision_score(
            y_true, y_prob
        )
    else:
        results["ROC_AUC"] = None
        results["PR_AUC"] = None

    return results
