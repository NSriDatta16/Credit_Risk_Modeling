from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "credit_risk_pipeline.joblib"

FEATURES = [
    "person_age",
    "person_income",
    "person_home_ownership",
    "person_emp_length",
    "loan_intent",
    "loan_grade",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_default_on_file",
    "cb_person_cred_hist_length",
]

_model = None


def get_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model artifact not found at {MODEL_PATH}. Run src/train.py first."
            )
        _model = joblib.load(MODEL_PATH)
    return _model


def predict(payload: dict) -> dict:
    missing = [feature for feature in FEATURES if feature not in payload]
    if missing:
        raise ValueError(f"Missing required fields: {', '.join(missing)}")

    row = {feature: payload[feature] for feature in FEATURES}
    frame = pd.DataFrame([row])
    model = get_model()
    probability = float(model.predict_proba(frame)[0, 1])
    prediction = int(probability >= 0.50)

    if probability >= 0.70:
        risk_level = "HIGH"
    elif probability >= 0.40:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    return {
        "default_prediction": prediction,
        "default_probability": round(probability, 4),
        "risk_level": risk_level,
    }
