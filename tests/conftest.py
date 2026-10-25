import pytest
from fastapi.testclient import TestClient

from src.app import main
from src.train.train import ROOT, train


@pytest.fixture(scope="session")
def model_path(tmp_path_factory):
    """Train a fresh model into a temp dir, so tests never depend on a stale file."""
    path = tmp_path_factory.mktemp("model") / "model.joblib"
    train(ROOT / "data" / "churn_v2.csv", path)
    return path


@pytest.fixture(scope="session")
def client(model_path):
    patch = pytest.MonkeyPatch()
    patch.setattr(main, "MODEL_PATH", model_path)
    with TestClient(main.app) as c:  # `with` runs the lifespan, which loads the model
        yield c
    patch.undo()


@pytest.fixture
def valid_customer():
    return {
        "tenure_months": 5,
        "monthly_fee": 89.9,
        "contract": "month-to-month",
        "internet_service": "fiber",
        "support_calls": 3,
    }
