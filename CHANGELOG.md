# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project aims to adhere to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Leakage-safe data ingestion for the UCI Diabetes 130-US Hospitals dataset
  (drop death/hospice discharges, normalise the `?` token, 30-day target,
  first-encounter-per-patient de-duplication).
- Configuration (Pydantic Settings), logging, domain constants, and a CC BY 4.0
  dataset download helper.
- Feature engineering: ICD-9 → clinical comorbidity grouping, prior-utilisation,
  medication complexity, A1C/glucose ordinals, age midpoint.
- Feature schema + `ColumnTransformer` preprocessing pipeline.
- Leakage-safe `StratifiedGroupKFold` cross-validation grouped on `patient_nbr`.
- Baseline class-weighted Logistic Regression pipeline with grouped-CV evaluation
  (AUROC / AUPRC) and a `python -m readmission.train` entry point.
- Tooling: Ruff, Black, MyPy (strict), Pytest, pre-commit, GitHub Actions CI
  (Python 3.11 & 3.12), `py.typed` marker, 100% test coverage (90% CI floor).

[Unreleased]: https://github.com/umairrana99/nhs-healthcare-ds-portfolio
