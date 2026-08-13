import os
import numpy as np
import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score

def train_and_save():
    # 1. Generate Synthetic Credit Default Dataset
    np.random.seed(42)
    n_samples = 2500

    data = pd.DataFrame({
        'age': np.random.randint(20, 65, size=n_samples),
        'income': np.random.randint(20000, 150000, size=n_samples),
        'loan_amount': np.random.randint(2000, 50000, size=n_samples),
        'credit_score': np.random.randint(300, 850, size=n_samples),
        'debt_to_income': np.random.uniform(0.1, 0.6, size=n_samples),
        'delinquencies': np.random.randint(0, 5, size=n_samples)
    })

    # Default condition logic
    score = (
        (data['debt_to_income'] * 3.5) - 
        (data['credit_score'] / 300.0) + 
        (data['loan_amount'] / data['income']) + 
        (data['delinquencies'] * 0.8)
    )
    data['default'] = (score > 1.2).astype(int)

    X = data.drop(columns=['default'])
    y = data['default']

    # 2. Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. Scaling Preprocessor
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. XGBoost Model Training
    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        eval_metric='logloss',
        random_state=42
    )
    model.fit(X_train_scaled, y_train)

    # 5. Validation
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC:  {roc_auc_score(y_test, y_prob):.4f}")

    # 6. Save Model Artifacts
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/xgboost_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    print("Model and Scaler successfully saved in 'models/' directory.")

if __name__ == '__main__':
    train_and_save()