<div align="center">

# 🏥 NHS Healthcare Data Science Portfolio

**Five production-quality healthcare machine-learning systems** — designed and built to the
standard expected of a Senior Data Scientist working with the NHS and similar healthcare organisations.

![Python](https://img.shields.io/badge/Python-3.12+-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?logo=scikit-learn&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-0194E2?logo=mlflow&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

</div>

---

> **Status:** 📐 Design phase. Each project begins with a complete design document in [`docs/`](docs/).
> Implementation starts only after all five design docs are reviewed. This README will grow as each
> project is delivered.

> ⚠️ **Data & ethics.** These projects use **public, de-identified research datasets**. No real
> patient-identifiable data is stored in this repository. Access-restricted datasets (MIMIC, n2c2,
> etc.) require their own credentialing and are **never redistributed here**. None of these systems
> is a certified medical device; they are research/portfolio work and not for clinical use.

## Projects

| # | Project | Type | Design Doc |
|---|---|---|---|
| 1 | **Hospital Readmission Prediction** | Tabular ML · classification | [`01`](docs/01-hospital-readmission-prediction.md) |
| 2 | **NHS Waiting List Analytics & Forecasting** | Time-series · analytics | [`02`](docs/02-nhs-waiting-list-analytics.md) |
| 3 | **ICU Mortality Risk Prediction** | Tabular ML · explainability | [`03`](docs/03-icu-mortality-prediction.md) |
| 4 | **Chest X-ray Pneumonia Detection** | Deep learning · vision · Grad-CAM | [`04`](docs/04-chest-xray-pneumonia-detection.md) |
| 5 | **Clinical NLP Information Extraction** | NLP · transformers · NER | [`05`](docs/05-clinical-nlp-system.md) |

## Engineering standards

Every project is built as if it were going to production — not as a single notebook:

- **Clean, modular architecture** with type hints, logging, and configuration management (Pydantic Settings)
- **REST API** (FastAPI) + **interactive dashboard** (Streamlit)
- **Experiment tracking** (MLflow) and reproducible pipelines
- **Testing** (Pytest — unit + integration) and **code quality** (Ruff, Black, MyPy, pre-commit)
- **Containerisation** (Docker / Docker Compose) and **CI/CD** (GitHub Actions)
- **Documentation** — professional README, architecture & data-flow diagrams, API examples

## Technology stack

**Language** Python 3.12+ · **ML** scikit-learn, XGBoost, LightGBM, CatBoost · **DL** PyTorch, TensorFlow ·
**NLP** spaCy/scispaCy, Transformers, ClinicalBERT/BioBERT · **Viz** Plotly, Matplotlib ·
**API** FastAPI · **Dashboard** Streamlit · **DB** PostgreSQL · **Tracking** MLflow ·
**Quality** Ruff, Black, MyPy, Pytest, pre-commit · **Infra** Docker, GitHub Actions

## Repository layout

```
nhs-healthcare-ds-portfolio/
├── docs/                     # Design documents (one per project) — start here
├── 01-hospital-readmission/  # (added during implementation)
├── 02-waiting-list-analytics/
├── 03-icu-mortality/
├── 04-pneumonia-detection/
├── 05-clinical-nlp/
├── LICENSE
└── README.md
```

## Author

**Umair Ali** — MSc Data Science, University of South Wales, UK
💼 [LinkedIn](https://www.linkedin.com/in/umair-ali99/) · 📧 ranaumairali.official@gmail.com
