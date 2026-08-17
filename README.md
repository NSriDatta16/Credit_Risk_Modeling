# Credit Risk Modeling using Machine Learning

End-to-end credit default prediction project covering exploratory analysis, feature engineering, model training, production inference, Flask API, Docker, and CI validation.

## Business Problem

Predict whether a borrower is likely to default on a loan. The target is `loan_status`:

- `0` → No Default
- `1` → Default

Dataset: 32,581 rows and 12 original columns.

## Production Architecture

```text
Applicant data
    ↓
Flask REST API / Web UI
    ↓
Saved scikit-learn Pipeline
    ↓
Preprocessing + model
    ↓
Default probability
    ↓
LOW / MEDIUM / HIGH risk
```

The Docker image trains the production artifact during image build so the container is self-contained and reproducible.

## Project Structure

```text
Credit_Risk_Modeling/
├── data/
│   └── credit_risk_dataset.csv
├── notebooks/
│   └── Credit_Risk_Modeling_End_To_End.ipynb
├── src/
│   ├── train.py
│   └── predictor.py
├── templates/
│   └── index.html
├── static/
│   └── style.css
├── main.py
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── requirements.txt
├── requirements-prod.txt
└── .github/workflows/docker.yml
```

## Training Pipeline

`src/train.py` performs production-oriented cleaning, feature engineering, stratified splitting, numeric/categorical preprocessing, and model comparison between balanced Logistic Regression and Random Forest. The best model by ROC-AUC is persisted as `models/credit_risk_pipeline.joblib`.

Engineered features include:

- `loan_to_income`
- `income_per_year_of_credit`
- `loan_rate_interaction`

## Run Locally

```bash
python -m pip install -r requirements-prod.txt
python src/train.py
python main.py
```

Open `http://localhost:5000`.

Health check:

```bash
curl http://localhost:5000/health
```

Prediction API:

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "person_age": 30,
    "person_income": 60000,
    "person_home_ownership": "RENT",
    "person_emp_length": 5,
    "loan_intent": "PERSONAL",
    "loan_grade": "B",
    "loan_amnt": 10000,
    "loan_int_rate": 12.5,
    "loan_percent_income": 0.17,
    "cb_person_default_on_file": "N",
    "cb_person_cred_hist_length": 7
  }'
```

## Docker

Build the image:

```bash
docker build -t credit-risk-model:latest .
```

Run it:

```bash
docker run --rm -p 5000:5000 credit-risk-model:latest
```

Or use Compose:

```bash
docker compose up --build
```

The image exposes port `5000` and includes a Docker health check.

## CI/CD

GitHub Actions builds the Docker image on pushes and pull requests and starts the container to validate the `/health` endpoint.

## Important Modeling Note

The probability returned by the API is a model estimate, not a lending decision. Production lending systems should additionally incorporate calibrated probabilities, approved risk policies, fairness testing, monitoring, human/underwriter controls, and regulatory requirements.
