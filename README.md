# 🏥 Hospital 30-Day Readmission Prediction

> Predict, at discharge, the probability that a patient is readmitted as an emergency within **30 days** —
> a calibrated, explainable decision-support tool. Framed with **NHS** definitions (NHS Outcomes Framework 3b)
> and trained on the openly-licensed **UCI Diabetes 130-US Hospitals** dataset.

![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-006600)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white)
![Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Why this project

Unplanned 30-day readmissions are an NHS quality indicator with a history of direct financial consequences
for providers (the Payment-by-Results non-payment rule, 2011/12). The goal is not to predict readmission for
its own sake, but to **rank patients so scarce discharge and community resources are targeted** where they help
most. The design foregrounds the things that make readmission models fail in practice:

- **Death/hospice discharges are removed** — those patients cannot be readmitted; leaving them in leaks the outcome.
- **Patient-level grouped cross-validation** — a patient recurs across encounters, so a random split leaks.
- **The correct 30-day target** — only `"<30"` is positive; folding in `">30"` answers a different question.
- **Calibration + decision-curve analysis** over raw accuracy — at ~11% prevalence a "never readmit" model is
  ~89% accurate and useless.

## What it does

- **Leakage-safe data pipeline** — cleaning, ICD-9 → clinical-group mapping, engineered features
  (prior utilisation, medication complexity, A1C/glucose, age), and a `ColumnTransformer` preprocessor.
- **Models** — a Logistic Regression baseline plus **XGBoost / LightGBM**, with probability **calibration**.
- **Honest evaluation** — grouped-CV **AUROC, AUPRC, Brier, ECE** and **decision-curve (net-benefit)** analysis.
- **Explainability** — **SHAP** per-patient contributions and global importance.
- **Serving** — a **FastAPI** service (`/predict`, `/explain`) and an interactive **Streamlit** dashboard.
- **Ops** — model persistence, **Docker/compose**, drift monitoring (PSI), and optional **MLflow** tracking.

## Quickstart

```bash
python -m pip install -e ".[dev]"          # install package + dev tooling
python -m readmission.data.download        # fetch the dataset (CC BY 4.0; not committed)
python -m readmission.train                # compare models, save the served model
make check                                 # ruff + black + mypy + pytest
```

Run the API and dashboard:

```bash
uvicorn readmission.api.app:app --reload            # API at http://localhost:8000/docs
streamlit run src/readmission/dashboard.py          # dashboard at http://localhost:8501
# or both together:
docker compose up --build
```

### API example

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{
  "time_in_hospital": 6, "num_lab_procedures": 45, "num_procedures": 1, "num_medications": 18,
  "number_outpatient": 0, "number_emergency": 1, "number_inpatient": 2, "number_diagnoses": 9,
  "gender": "Female", "admission_type_id": 1, "discharge_disposition_id": 1,
  "admission_source_id": 7, "diag_1": "410", "age": "[70-80)", "insulin": "Up", "A1Cresult": ">8"
}'
```

```json
{ "readmission_probability": 0.31, "risk_band": "High",
  "disclaimer": "Decision support only; not a substitute for clinical judgement." }
```

`POST /explain` returns the same probability plus the top signed SHAP contributions.

## Project structure

```
src/readmission/
├── config.py · constants.py · logging_utils.py
├── data/          # download + leakage-safe ingestion
├── features/      # ICD-9 grouping, engineered features, schema
├── preprocess.py  # ColumnTransformer
├── cv.py          # StratifiedGroupKFold on patient_nbr
├── models/        # baseline, boosted, calibration
├── metrics.py     # ECE, decision-curve net benefit
├── evaluate.py    # grouped-CV + model comparison
├── explain.py     # SHAP
├── service.py     # shared scoring (API + dashboard)
├── api/           # FastAPI app
├── dashboard.py   # Streamlit
├── monitoring.py  # drift (PSI)
├── tracking.py    # optional MLflow
└── train.py       # entry point
tests/             # 80+ tests, 100% coverage (grouped-leakage tests included)
Dockerfile · docker-compose.yml · .github/workflows/ci.yml
```

## Tech stack

Python 3.11+ · pandas · scikit-learn · XGBoost · LightGBM · SHAP · FastAPI · Streamlit · joblib ·
MLflow (optional) · Docker · pytest · Ruff · Black · MyPy · GitHub Actions.

## Dataset & limitations

**UCI Diabetes 130-US Hospitals (1999–2008)** — 101,766 encounters, ~47 features, **CC BY 4.0**.
Target `readmitted ∈ {"<30", ">30", "NO"}` → binary `readmitted_30d` (`"<30"` = 1), ~11% positive.
[Dataset page](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008).

> Data files are **gitignored** and never committed. The model is trained on US data and is **not** validated
> for any live NHS population — this is a portfolio/decision-support demonstrator, **not a medical device**.

## Author

**Umair Ali** — MSc Data Science, University of South Wales, UK · [LinkedIn](https://www.linkedin.com/in/umair-ali99/)
