# 🏥 Hospital 30-Day Readmission Prediction

> Predict, at discharge, the probability that a patient is readmitted as an emergency within **30 days** —
> a calibrated, explainable decision-support tool. Framed with **NHS** definitions (NHS Outcomes Framework 3b),
> trained on the openly-licensed **UCI Diabetes 130-US Hospitals** dataset.

![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?logo=scikit-learn&logoColor=white)
![Ruff](https://img.shields.io/badge/lint-ruff-261230?logo=ruff&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

> 🚧 **Status: early development (~20–30%).** The data-ingestion layer, configuration, tests and CI are in
> place. Feature engineering, modelling, calibration, API and dashboard are next — see the roadmap below.

---

## Why this project

Unplanned 30-day readmissions are an NHS quality indicator with a history of direct financial consequences for
providers (Payment-by-Results non-payment rule, 2011/12). The goal is not to predict readmission for its own
sake, but to **rank patients so scarce discharge and community resources are targeted** where they help most.

The build deliberately foregrounds the things that make readmission models fail in practice:

- **Removing death/hospice discharges** — those patients cannot be readmitted; leaving them in leaks the outcome.
- **Patient-level grouped splitting** — a patient recurs across encounters, so random splits leak.
- **Correct 30-day target** — only `"<30"` is positive; folding in `">30"` answers a different question.
- **Calibration + net-benefit** over raw accuracy (an ~89%-accurate "never readmit" model is useless).

## What's implemented so far

| Area | Status |
|---|---|
| Package scaffold, `pyproject`, `Makefile` | ✅ |
| Config management (Pydantic Settings, env-overridable) | ✅ |
| Logging setup | ✅ |
| Dataset download helper (CC BY 4.0, not committed) | ✅ |
| Data ingestion + cleaning (missing tokens, death/hospice removal, target, dedup) | ✅ |
| Unit tests incl. **leakage tests** | ✅ |
| CI (Ruff, Black, MyPy, Pytest × Python 3.11/3.12) + pre-commit | ✅ |
| Feature engineering / preprocessing | ⏳ next |
| Baseline (LogReg) → boosters, calibration, SHAP | ⏳ |
| FastAPI service + Streamlit dashboard | ⏳ |
| MLflow tracking, Docker, monitoring | ⏳ |

## Getting started

```bash
python -m pip install -e ".[dev]"     # or: make install

# Fetch the dataset (CC BY 4.0; ~a few MB). Not committed to the repo.
python -m readmission.data.download

# Quality gates
make check    # ruff + mypy + pytest
```

Load and clean the data:

```python
from readmission.config import get_settings
from readmission.data.ingest import load_raw, build_dataset

settings = get_settings()
df = build_dataset(load_raw(settings.raw_data_path), first_encounter_only=True)
```

## Dataset

**UCI Diabetes 130-US Hospitals (1999–2008)** — 101,766 encounters, ~47 features, **CC BY 4.0**.
Target `readmitted ∈ {"<30", ">30", "NO"}` → binary `readmitted_30d` (`"<30"` = 1). ~11% positive.
Download: <https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008>.

> Data files are **gitignored** and never committed. The model is trained on US data and is **not** validated
> for any live NHS population — it is a portfolio/decision-support demonstrator, not a medical device.

## Layout

```
.
├── src/readmission/
│   ├── config.py          # Pydantic Settings
│   ├── constants.py       # domain constants (death/hospice codes, target, columns)
│   ├── logging_utils.py
│   └── data/
│       ├── download.py     # fetch the CC BY 4.0 dataset
│       └── ingest.py       # load + clean (leakage-safe)
├── tests/                  # unit + leakage tests
├── data/raw/               # gitignored — place diabetic_data.csv here
├── .github/workflows/      # CI (ruff, black, mypy, pytest)
├── pyproject.toml ├─ Makefile └─ .pre-commit-config.yaml
```

## Roadmap

1. Feature engineering (prior-utilisation, comorbidity grouping of ICD-9, medication complexity, A1C).
2. `ColumnTransformer` preprocessing + `StratifiedGroupKFold` on `patient_nbr`.
3. Baseline Logistic Regression → XGBoost / LightGBM / CatBoost (Optuna), probability calibration.
4. Evaluation: AUPRC, calibration (Brier/ECE), decision-curve analysis; SHAP explainability; fairness slices.
5. FastAPI `/predict` + Streamlit dashboard; MLflow tracking; Docker; drift monitoring.

## Author

**Umair Ali** — MSc Data Science, University of South Wales, UK · [LinkedIn](https://www.linkedin.com/in/umair-ali99/)
