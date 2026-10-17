"""Train and compare candidate default models without replacing deployment artifacts."""

import argparse
import json
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.evaluate import evaluate_model
from src.feature_engineering import add_credit_features

ROOT = Path(__file__).resolve().parents[1]
TARGET = "default.payment.next.month"
EXTRA_FEATURES = [
    *(f"UTILIZATION_{i}" for i in range(1, 7)),
    "AVG_UTILIZATION", "MAX_UTILIZATION", "AVG_DELAY", "MAX_DELAY",
    "DELAY_MONTHS", "AVG_BILL_AMT", "AVG_PAY_AMT",
    "PAY_TO_BILL_RATIO", "PAY_TO_LIMIT_RATIO",
]
CATEGORICAL = ["SEX", "EDUCATION", "MARRIAGE"]


def _raw_features():
    return json.loads((ROOT / "config/input_features.json").read_text(encoding="utf-8"))


def validate_training_data(df):
    raw = _raw_features()
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("Training data must be a nonempty DataFrame.")
    missing = set(raw + [TARGET]) - set(df.columns)
    if missing:
        raise ValueError(f"Missing training columns: {sorted(missing)}")
    clean = df.loc[:, raw + [TARGET]].copy()
    for col in clean.columns:
        clean[col] = pd.to_numeric(clean[col], errors="raise")
    if not np.isfinite(clean.to_numpy(dtype=float)).all():
        raise ValueError("Dataset includes missing or infinite values.")
    if (clean["LIMIT_BAL"] <= 0).any():
        raise ValueError("LIMIT_BAL must be positive.")
    if not clean[TARGET].isin([0, 1]).all() or clean[TARGET].nunique() != 2:
        raise ValueError("Target must include both classes 0 and 1.")
    if clean[TARGET].value_counts().min() < 8:
        raise ValueError("Too few samples in one class for stratified evaluation.")
    return clean, raw


def candidate_specs(seed=42):
    return {
        "logistic_regression": (
            LogisticRegression(max_iter=2000, random_state=seed),
            {"classifier__C": [0.1, 1.0, 3.0],
             "classifier__class_weight": [None, "balanced"]},
        ),
        "random_forest": (
            RandomForestClassifier(random_state=seed, n_jobs=1),
            {"classifier__n_estimators": [100, 200],
             "classifier__max_depth": [8, 12, None],
             "classifier__min_samples_leaf": [2, 5],
             "classifier__class_weight": [None, "balanced"]},
        ),
        "hist_gradient_boosting": (
            HistGradientBoostingClassifier(random_state=seed),
            {"classifier__max_iter": [100, 180],
             "classifier__learning_rate": [0.05, 0.1],
             "classifier__max_leaf_nodes": [15, 31]},
        ),
    }


def train_candidate(df, output_dir, *, folds=3, iterations=3, seed=42):
    """Choose best candidate by CV average precision, then score untouched test set."""
    if not isinstance(folds, int) or folds < 2:
        raise ValueError("folds must be at least 2.")
    if not isinstance(iterations, int) or iterations < 1:
        raise ValueError("iterations must be at least 1.")

    clean, raw = validate_training_data(df)
    features = raw + EXTRA_FEATURES
    train_df, test_df = train_test_split(
        clean, test_size=0.20, stratify=clean[TARGET], random_state=seed
    )
    y_train = train_df[TARGET].astype(int)
    y_test = test_df[TARGET].astype(int)
    if y_train.value_counts().min() < folds:
        raise ValueError("Not enough examples of each class for CV folds.")

    # Each engineered feature uses only the current customer's own fields.
    X_train = add_credit_features(train_df[raw]).loc[:, features]
    X_test = add_credit_features(test_df[raw]).loc[:, features]

    numeric = [name for name in features if name not in CATEGORICAL]
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
    ])
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    leaderboard = []
    best = None

    for name, (classifier, search_space) in candidate_specs(seed).items():
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ])
        search = RandomizedSearchCV(
            pipeline, search_space, n_iter=iterations,
            scoring={"average_precision": "average_precision", "roc_auc": "roc_auc"},
            refit="average_precision", cv=cv, n_jobs=1,
            random_state=seed, error_score="raise",
        )
        search.fit(X_train, y_train)
        record = {
            "algorithm": name,
            "cv_average_precision": float(search.best_score_),
            "cv_roc_auc": float(search.cv_results_["mean_test_roc_auc"][search.best_index_]),
            "best_parameters": search.best_params_,
        }
        leaderboard.append(record)
        if best is None or record["cv_average_precision"] > best[0]:
            best = (record["cv_average_precision"], name, search.best_estimator_)

    _, winner_name, winning_pipeline = best
    threshold = 0.5  # A documented baseline; NOT optimized on the test set.
    probability = winning_pipeline.predict_proba(X_test)[:, 1]
    result = {
        "dataset_rows": int(len(clean)),
        "train_rows": int(len(train_df)),
        "heldout_test_rows": int(len(test_df)),
        "feature_count": len(features),
        "cv_folds": folds,
        "search_iterations_per_algorithm": iterations,
        "selection_metric": "cross_validation_average_precision",
        "selected_algorithm": winner_name,
        "comparison": sorted(leaderboard, key=lambda v: v["cv_average_precision"], reverse=True),
        "heldout_test": evaluate_model(y_test.to_numpy(), probability, threshold=threshold),
        "limitation": "Candidate requires calibration, fairness testing, and release validation before deployment.",
    }
    output = Path(output_dir)
    artifact_path = output / "credit_default_candidate.joblib"
    report_path = output / "candidate_evaluation.json"
    if artifact_path.exists() or report_path.exists():
        raise FileExistsError("Candidate files already exist: choose a new output directory.")
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "pipeline": winning_pipeline,
        "threshold": threshold,
        "model_name": winner_name,
        "model_version": "candidate-1",
        "features": features,
        "target": TARGET,
    }, artifact_path)
    report_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def synthetic_data(n=120, seed=42):
    """Generate synthetic customers ONLY for a smoke test; never for deployment."""
    rng = np.random.default_rng(seed)
    raw = _raw_features()
    records = pd.DataFrame(index=range(n), columns=raw, dtype=float)
    records["LIMIT_BAL"] = rng.integers(20000, 500001, n)
    records["SEX"] = rng.choice([1, 2], n)
    records["EDUCATION"] = rng.choice([1, 2, 3, 4], n)
    records["MARRIAGE"] = rng.choice([1, 2, 3], n)
    records["AGE"] = rng.integers(21, 71, n)
    for t in [0, 2, 3, 4, 5, 6]:
        records[f"PAY_{t}"] = rng.choice([-2, -1, 0, 1, 2, 3], size=n)
    for i in range(1, 7):
        records[f"BILL_AMT{i}"] = rng.integers(0, 120001, size=n)
        records[f"PAY_AMT{i}"] = rng.integers(0, 50001, size=n)
    risk = 0.1 + 0.12 * (records["PAY_0"] > 0) + 0.08 * (records["PAY_2"] > 0)
    records[TARGET] = rng.binomial(1, risk.clip(0.01, 0.95).to_numpy())
    return records


def main():
    parser = argparse.ArgumentParser(description="Train NON-DEPLOYED credit default candidate.")
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--csv", type=Path, help="Local, labeled UCI credit dataset CSV")
    choice.add_argument("--smoke-test", action="store_true", help="Use synthetic records for pipeline testing only")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "models" / "candidates")
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=3)
    args = parser.parse_args()

    if args.smoke_test:
        with tempfile.TemporaryDirectory() as temp:
            results = train_candidate(synthetic_data(), temp, folds=2, iterations=1)
        print("SMOKE TEST ONLY - SYNTHETIC DATA, NOT MODEL PERFORMANCE")
    else:
        results = train_candidate(
            pd.read_csv(args.csv), args.output_dir,
            folds=args.folds, iterations=args.iterations,
        )
        print("Candidate saved outside deployed model path.")
    print(json.dumps({
        "selected_algorithm": results["selected_algorithm"],
        "feature_count": results["feature_count"],
        "algorithms_compared": len(results["comparison"]),
        "heldout_test": results["heldout_test"],
    }, indent=2))


if __name__ == "__main__":
    main()
