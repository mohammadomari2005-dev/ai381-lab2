# AI381 Lab 2: Churn prediction service

[![CI](https://github.com/mohammadomari2005-dev/ai381-lab2/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/mohammadomari2005-dev/ai381-lab2/actions/workflows/ci.yml)

A customer-churn model (scikit-learn logistic regression) served by FastAPI, packaged as a
multi-stage, non-root Docker image, and built, scanned (Trivy) and published to GHCR by
GitHub Actions on every commit to `main`.

## Run it (one command)

```bash
docker run --rm -p 8000:8000 ghcr.io/mohammadomari2005-dev/ai381-lab2@sha256:767dd41f2c019e2c792c22f840d0b911476dab9228ff6dae36365d9076da11b0
```

Then, in another terminal:

```bash
curl -s localhost:8000/health

curl -s -X POST localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"tenure_months":5,"monthly_fee":89.9,"contract":"month-to-month","internet_service":"fiber","support_calls":3}'
```

Interactive docs: http://localhost:8000/docs

Inputs are validated with pydantic: wrong types, unknown categories, out-of-range values or
unexpected fields (such as the old `monthly_charges` name) return HTTP 422.

## Develop locally

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
python src/train/train.py        # trains and saves model/model.joblib
pytest -v                        # 10 tests
ruff check .                     # lint
uvicorn src.app.main:app --port 8000
```

## Project layout

| Path | What it is |
|---|---|
| `src/train/train.py` | Trains the model and writes `model/model.joblib` and `metrics.json` |
| `src/app/main.py` | FastAPI app: `/health` and `/predict` |
| `tests/` | pytest suite: API contract, model loading, input schema, edge cases, determinism |
| `Dockerfile` | Multi-stage, non-root, pinned base image digest, HEALTHCHECK |
| `.github/workflows/ci.yml` | CI: lint and test on PRs; build, Trivy scan, push to GHCR on `main` |

## CI

- **Pull requests:** `lint-and-test` (ruff + pytest), then `build-scan-push` builds the image and
  scans it with Trivy (HIGH/CRITICAL fail the build). Nothing is pushed from a PR.
- **Push to `main`:** the same checks, then the image is pushed to
  `ghcr.io/mohammadomari2005-dev/ai381-lab2` tagged with the commit SHA.
- Every third-party action is pinned to a full commit SHA.
- `main` is protected: changes go through a pull request with both checks passing.

