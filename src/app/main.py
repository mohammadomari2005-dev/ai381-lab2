"""Churn prediction API (FastAPI).

Run from the project root:
    uvicorn src.app.main:app --port 8000
"""
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

DEFAULT_MODEL = Path(__file__).resolve().parents[2] / "model" / "model.joblib"
MODEL_PATH = Path(os.getenv("MODEL_PATH", DEFAULT_MODEL))
FEATURES = ["tenure_months", "monthly_fee", "support_calls", "contract", "internet_service"]


class Customer(BaseModel):
    # extra="forbid": unknown fields (e.g. the old `monthly_charges`) are rejected with 422
    model_config = ConfigDict(extra="forbid")

    tenure_months: int = Field(ge=0, le=120)
    monthly_fee: float = Field(ge=0, le=1000)
    support_calls: int = Field(ge=0, le=100)
    contract: Literal["month-to-month", "one-year", "two-year"]
    internet_service: Literal["dsl", "fiber", "none"]


class Prediction(BaseModel):
    churn_probability: float
    churn_prediction: bool


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Fail fast: if the model file is missing, the service refuses to start.
    app.state.model = joblib.load(MODEL_PATH)
    yield


app = FastAPI(title="AI381 Lab 2 churn API", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=Prediction)
def predict(customer: Customer):
    X = pd.DataFrame([customer.model_dump()])[FEATURES]
    p = float(app.state.model.predict_proba(X)[0, 1])
    return Prediction(churn_probability=round(p, 3), churn_prediction=p >= 0.5)
