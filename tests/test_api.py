import joblib
import pandas as pd
import pytest

from src.train.train import ROOT, train


# 1. API contract: status codes and response shape
def test_api_contract(client, valid_customer):
    assert client.get("/health").json() == {"status": "ok"}

    r = client.post("/predict", json=valid_customer)
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"churn_probability", "churn_prediction"}
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert isinstance(body["churn_prediction"], bool)


# 2. Model loads: the service starts with a real, usable model
def test_model_loads(client):
    model = client.app.state.model
    assert hasattr(model, "predict_proba")
    assert list(model.feature_names_in_) == [
        "tenure_months", "monthly_fee", "support_calls", "contract", "internet_service",
    ]


# 3. Input schema: bad inputs are rejected with 422, never a 500
@pytest.mark.parametrize(
    "change",
    [
        {"contract": "weekly"},            # category the model never saw
        {"tenure_months": "five"},         # wrong type
        {"support_calls": -1},             # out of range
        {"monthly_charges": 89.9},         # old v1 column name (extra field)
    ],
)
def test_invalid_input_returns_422(client, valid_customer, change):
    payload = {**valid_customer, **change}
    assert client.post("/predict", json=payload).status_code == 422


def test_missing_field_returns_422(client, valid_customer):
    del valid_customer["monthly_fee"]
    assert client.post("/predict", json=valid_customer).status_code == 422


# 4. Edge cases: boundary values are accepted, just beyond them rejected
def test_edge_cases(client, valid_customer):
    low = {**valid_customer, "tenure_months": 0, "monthly_fee": 0, "support_calls": 0}
    high = {**valid_customer, "tenure_months": 120, "monthly_fee": 1000, "support_calls": 100}
    for payload in (low, high):
        r = client.post("/predict", json=payload)
        assert r.status_code == 200
        assert 0.0 <= r.json()["churn_probability"] <= 1.0

    assert client.post("/predict", json={**valid_customer, "tenure_months": 121}).status_code == 422


# 5. Deterministic prediction: same input -> same output, and same seed -> same model
def test_prediction_is_deterministic(client, valid_customer, tmp_path):
    first = client.post("/predict", json=valid_customer).json()
    second = client.post("/predict", json=valid_customer).json()
    assert first == second

    data = ROOT / "data" / "churn_v2.csv"
    train(data, tmp_path / "a.joblib")
    train(data, tmp_path / "b.joblib")
    X = pd.DataFrame([valid_customer])
    a = joblib.load(tmp_path / "a.joblib").predict_proba(X)
    b = joblib.load(tmp_path / "b.joblib").predict_proba(X)
    assert (a == b).all()


# 6. Sanity: a risky customer scores higher than a loyal one
def test_risky_customer_scores_higher(client, valid_customer):
    risky = {**valid_customer, "tenure_months": 2, "support_calls": 6, "contract": "month-to-month"}
    loyal = {**valid_customer, "tenure_months": 60, "support_calls": 0, "contract": "two-year"}
    p_risky = client.post("/predict", json=risky).json()["churn_probability"]
    p_loyal = client.post("/predict", json=loyal).json()["churn_probability"]
    assert p_risky > p_loyal
