import json
import joblib
import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder
from src.preprocessing import prepare_model_input

DATA = r"C:\Users\mianh\Downloads\UCI_Credit_Card.csv"

df = pd.read_csv(DATA)

artifact = joblib.load(
    "models/credit_default_model_v1.joblib"
)

with open("config/input_features.json") as f:
    raw_features = json.load(f)

X = prepare_model_input(
    df[raw_features],
    artifact["features"]
)

preprocessor = artifact["pipeline"].named_steps["preprocessor"]

print("\nENCODER CATEGORY AUDIT")
print("=" * 50)

found = False

for name, transformer, columns in preprocessor.transformers_:

    if isinstance(transformer, OneHotEncoder):
        encoder = transformer
    elif hasattr(transformer, "steps"):
        encoder = next(
            (
                step for _, step in transformer.steps
                if isinstance(step, OneHotEncoder)
            ),
            None
        )
    else:
        encoder = None

    if encoder is None:
        continue

    found = True

    if isinstance(columns, str):
        column_names = [columns]
    elif isinstance(columns, slice):
        column_names = list(X.columns[columns])
    else:
        column_names = [
            X.columns[int(c)]
            if isinstance(c, (int, np.integer))
            else str(c)
            for c in columns
        ]

    for col, known in zip(column_names, encoder.categories_):

        print(f"\nCOLUMN: {col}")
        print("Known categories:", list(known))

        unknown = X.loc[
            ~X[col].isin(known),
            col
        ]

        print("Unknown records:", len(unknown))

        if len(unknown):
            print("Unknown category counts:")
            print(unknown.value_counts(dropna=False))

if not found:
    print("No fitted OneHotEncoder was found.")

print("\nAudit completed. No model or data changed.")
