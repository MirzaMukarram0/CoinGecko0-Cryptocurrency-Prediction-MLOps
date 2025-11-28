# Git Workflow and CI/CD Pipeline Documentation

## Overview

This project implements a complete CI/CD pipeline using GitHub Actions with CML (Continuous Machine Learning) integration for automated model performance reporting.

## Branching Model

```
feature/* ──► dev ──► test ──► main/master
     │          │        │         │
     │          │        │         └─ Production Deployment
     │          │        └─ Model Retraining + CML Reports
     │          └─ Code Quality + Unit Tests
     └─ Feature Development
```

## CI/CD Pipeline Summary

| Merge Event | CI Action | CML Integration |
|-------------|-----------|-----------------|
| **feature → dev** | Run Code Quality Checks (Linting) and Unit Tests | N/A |
| **dev → test** | Model Retraining Test: Trigger full data and model training pipeline | CML generates metric comparison report in PR comments. Merge blocked if new model performs worse. |
| **test → main** | Full Production Deployment Pipeline | N/A |

---

## Workflow Details

### 1. Dev Branch: Code Quality & Unit Tests

**File:** `.github/workflows/dev-code-quality.yml`

**Triggers:** Push/PR to `dev` branch, pushes to `feature/**` branches

**Jobs:**
- **Code Quality Analysis**
  - Flake8 linting (critical errors block merge)
  - Black code formatting check
  - isort import sorting check
  - Bandit security scan

- **Unit Tests**
  - Pytest with coverage
  - Test reports uploaded as artifacts

### 2. Test Branch: Model Retraining with CML

**File:** `.github/workflows/test-model-retraining.yml`

**Triggers:** Push/PR to `test` branch

**Jobs:**
- **Model Retraining Pipeline**
  - Extract fresh data from CoinGecko API
  - Transform and prepare features
  - Train models (RandomForest, GradientBoosting, Ridge)
  - Compare with production model metrics

- **CML Report Generation** (PR only)
  - Generate metric comparison charts
  - Post automated report as PR comment
  - **Block merge if new model performs worse**

- **Integration Tests**
  - Validate module imports
  - DAG syntax validation

### 3. Main Branch: Production Deployment

**File:** `.github/workflows/main-deploy.yml`

**Triggers:** Push/PR to `main`/`master` branch

**Jobs:**
- **Final Validation**
  - Code quality check
  - All tests execution

- **Fetch Best Model from MLflow**
  - Query MLflow Model Registry
  - Get production or best-performing model

- **Build Docker Image**
  - Create Dockerfile for FastAPI
  - Build and push to Docker Hub
  - Tag with version (e.g., `v1.0.{run_number}`)

- **Deployment Verification**
  - Run container locally
  - Health check verification
  - API endpoint testing

---

## Docker Containerization

The model is served via a FastAPI REST API inside a Docker container.

### Dockerfile (Auto-generated in CI)

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt uvicorn[standard]
COPY src/ ./src/
ENV PYTHONPATH=/app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s CMD curl -f http://localhost:8000/health || exit 1
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Running Locally

```bash
# Build image
docker build -t crypto-prediction-api:latest -f Dockerfile.api .

# Run container
docker run -d -p 8000:8000 --name crypto-api crypto-prediction-api:latest

# Check health
curl http://localhost:8000/health

# Access API docs
open http://localhost:8000/docs
```

---

## Required GitHub Secrets

Configure these in **Settings → Secrets and variables → Actions**:

| Secret | Description |
|--------|-------------|
| `MLFLOW_TRACKING_URI` | DagHub MLflow tracking URL |
| `MLFLOW_TRACKING_USERNAME` | DagHub username |
| `MLFLOW_TRACKING_PASSWORD` | DagHub token |
| `DOCKERHUB_USERNAME` | Docker Hub username |
| `DOCKERHUB_TOKEN` | Docker Hub access token |
| `GITHUB_TOKEN` | Auto-provided by GitHub (for CML) |

---

## Branch Protection Rules

### Dev Branch

```
Branch pattern: dev
✅ Require pull request before merging (1 approval)
✅ Require status checks: code-quality, unit-tests
✅ Require conversation resolution
```

### Test Branch

```
Branch pattern: test
✅ Require pull request before merging (1 approval)
✅ Require status checks: model-retraining, cml-report, integration-tests
✅ Require conversation resolution
```

### Main Branch

```
Branch pattern: main
✅ Require pull request before merging (2 approvals)
✅ Require status checks: final-validation, build-docker, deployment-verification
✅ Require conversation resolution
✅ Do not allow bypassing
```

---

## CML Report Example

When a PR is created from `dev` to `test`, CML automatically posts a comment like:

```markdown
# 🤖 Model Performance Comparison Report

## Summary
✅ **New model PASSED** - Ready for merge

## Metrics Comparison

| Metric | Production Model | New Model | Change |
|--------|-----------------|-----------|--------|
| RMSE | 1053.72 | 1021.45 | 🟢 -3.06% |
| MAE | 812.34 | 798.21 | 🟢 -1.74% |
| R² | 0.9234 | 0.9312 | 🟢 +0.84% |

## Model Details
**Best Model Type:** ridge

![Metrics Comparison](./metrics_comparison.png)
```

---

## Quick Commands Reference

```bash
# Create feature branch
git checkout dev && git pull
git checkout -b feature/my-feature

# Push and create PR to dev
git push -u origin feature/my-feature
# Create PR: feature/my-feature → dev

# After dev merge, create PR to test
# PR: dev → test (triggers CML report)

# After test passes, create PR to main
# PR: test → main (triggers deployment)
```

---

## Troubleshooting

### CML Report Not Appearing

1. Ensure `GITHUB_TOKEN` has write permissions
2. Check CML step logs for errors
3. Verify `iterative/setup-cml@v2` is installed

### Docker Build Failing

1. Check `DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN` secrets
2. Verify Dockerfile syntax
3. Check requirements.txt is up to date

### Model Comparison Failing

1. Ensure MLflow credentials are correct
2. Check if production model exists in registry
3. Review model training logs
