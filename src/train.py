from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "credit_risk_dataset.csv"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / "credit_risk_pipeline.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"

TARGET = "loan_status"


df = pd.read_csv(DATA_PATH)
df = df.drop_duplicates().copy()

df = df[df["person_age"].between(18, 100)].copy()
df = df[df["person_emp_length"].isna() | df["person_emp_length"].between(0, 60)].copy()

df["loan_to_income"] = df["loan_amnt"] / df["person_income"].replace(0, pd.NA)
df["income_per_year_of_credit"] = df["person_income"] / df["cb_person_cred_hist_length"].replace(0, pd.NA)
df["loan_rate_interaction"] = df["loan_percent_income"] * df["loan_int_rate"]

df = df.replace([float("inf"), float("-inf")], pd.NA)

X = df.drop(columns=[TARGET])
y = df[TARGET].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features),
])

models = {
    "logistic_regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
    "random_forest": RandomForestClassifier(
        n_estimators=400,
        max_depth=14,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    ),
}

results = {}
best_name = None
best_auc = -1.0
best_pipeline = None

for name, estimator in models.items():
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", estimator),
    ])
    pipeline.fit(X_train, y_train)
    probabilities = pipeline.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.50).astype(int)
    auc = roc_auc_score(y_test, probabilities)
    f1 = f1_score(y_test, predictions)
    accuracy = accuracy_score(y_test, predictions)
    results[name] = {
        "roc_auc": float(auc),
        "f1": float(f1),
        "accuracy": float(accuracy),
    }
    if auc > best_auc:
        best_auc = auc
        best_name = name
        best_pipeline = pipeline

joblib.dump(best_pipeline, MODEL_PATH)

import json
metrics = {
    "best_model": best_name,
    "test_metrics": results[best_name],
    "all_models": results,
    "features": X.columns.tolist(),
    "numeric_features": numeric_features,
    "categorical_features": categorical_features,
}
METRICS_PATH.write_text(json.dumps(metrics, indent=2))

print(f"Best model: {best_name}")
print(json.dumps(results, indent=2))
print("Saved:", MODEL_PATH)
