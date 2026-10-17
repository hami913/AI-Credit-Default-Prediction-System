import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("http://127.0.0.1:5000")

experiment_name = "Credit_Risk_Historical_Model_Comparison"
mlflow.set_experiment(experiment_name)

# Historical, previously reported Colab results.
# NOT new training runs or independently reverified scores.

models = [
    {
        "name": "Logistic_Regression_Baseline",
        "algorithm": "Logistic Regression",
        "Accuracy": 0.8088,
        "Precision": 0.6923,
        "Recall": 0.2442,
        "F1": 0.3610,
        "ROC_AUC": 0.7101,
        "PR_AUC": 0.4950
    },
    {
        "name": "Regularized_Random_Forest",
        "algorithm": "Random Forest",
        "Accuracy": 0.8165,
        "Precision": 0.6619,
        "Recall": 0.3482,
        "F1": 0.4563,
        "ROC_AUC": 0.7756,
        "PR_AUC": 0.5536
    },
    {
        "name": "Previously_Reported_Final_RF",
        "algorithm": "Random Forest",
        "Accuracy": 0.7768,
        "Precision": 0.4962,
        "Recall": 0.5863,
        "F1": 0.5375,
        "ROC_AUC": 0.7777,
        "PR_AUC": 0.5548
    }
]

client = MlflowClient()

experiment = client.get_experiment_by_name(experiment_name)

for model in models:
    existing = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=(
            "tags.historical_import_key = "
            f"'{model['name']}'"
        )
    )

    if existing:
        print("Already imported:", model["name"])
        continue

    with mlflow.start_run(run_name=model["name"]):
        mlflow.log_param("algorithm", model["algorithm"])
        mlflow.log_param("threshold", 0.50)

        for metric in [
            "Accuracy", "Precision", "Recall",
            "F1", "ROC_AUC", "PR_AUC"
        ]:
            mlflow.log_metric(metric, model[metric])

        mlflow.set_tags({
            "run_type": "historical_import",
            "historical_import_key": model["name"],
            "source": "previously_user_reported_colab_results",
            "verification": "not_independently_verified",
            "training_executed_in_this_run": "false",
            "saved_model_linkage": "not_verified"
        })

        print("Imported historical results:", model["name"])

print("\nHISTORICAL MLFLOW IMPORT COMPLETE")
print("Original trained model unchanged.")
