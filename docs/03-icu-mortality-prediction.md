# 03 — ICU Mortality Risk Prediction System

> **Design document.** Predict in-hospital mortality from the **first 24 hours** of an adult ICU admission and
> **explain every prediction** in clinically intelligible terms. Explainability (SHAP) is the centrepiece, not
> an add-on. Built on the open **WiDS Datathon 2020** data, with MIMIC-IV/eICU as credentialed extensions.

---

## Executive Summary

Severity-of-illness scoring (APACHE, SOFA, SAPS-II; in the UK, ICNARC's ICNARCH-2018) is decades-old ICU
practice, but hand-calibrated additive scores lose discrimination and drift out of calibration on modern
case-mix. A calibrated gradient-boosted model is, in effect, a higher-resolution, locally-recalibrated successor
— *provided* its output is a trustworthy probability and each prediction is explainable. This project builds
exactly that: WiDS 2020 (openly downloadable via Kaggle, ~131k stays, ships with APACHE IVa probabilities as a
free clinical baseline), calibrated boosters, per-patient TreeSHAP explanations, and honest subgroup/calibration
reporting positioned relative to ICNARC. It is decision-support and audit augmentation — never an automated
treatment or resource-denial trigger.

## Problem Statement

Given first-24h vitals, labs, demographics, admission context and comorbidities, output a calibrated probability
of in-hospital death (~8–9% prevalence) plus a signed, per-feature explanation. Success = **beat APACHE IVa on
the same folds** in discrimination *and* calibration, with subgroup fairness documented.

## NHS Relevance

UK adult critical care units submit to the **ICNARC Case Mix Programme**, which computes risk-adjusted predicted
mortality (ICNARCH-2018) and reports each unit's **Standardised Mortality Ratio (SMR = observed/expected)** for
national benchmarking. A production model must be positioned relative to ICNARCH-2018 — augmenting audit and
situational awareness, not competing with the official metric — and would sit under **DCB0129/DCB0160** clinical
safety and MHRA SaMD governance if used to influence care.

## Business Value

Primarily **quality, safety and audit**: earlier identification of deteriorating patients who look deceptively
stable, prioritised outreach review, aggregate acuity for staffing/bed-flow, and stronger risk-adjusted
governance evidence. Not a headcount-reduction tool.

## Expected Users

Intensivists and bedside clinicians (a calibrated "second read" + driver explanation), ICU outreach/rapid-response
teams (ranked review list), capacity planners (population acuity), and clinical-audit teams (SMR-style benchmarking).

## Stakeholders

| Stakeholder | Value | Needs from model |
|---|---|---|
| Intensivists | Early quantified risk, second read | Calibrated probability + local SHAP |
| Outreach / RRT | Prioritisation across the unit | Ranked list, stable thresholds |
| Bed management | Expected acuity → staffing | Population risk distribution/trend |
| Family communication | Structured, honest prognosis (never a "death date") | Interpretable drivers + uncertainty |
| Clinical audit (ICNARC-style) | Risk-adjusted mortality, QI | Calibration, subgroup fairness, documented method |
| IG / Caldicott | Lawful, de-identified use | DUA compliance, audit trail |

## Dataset Options

1. **WiDS Datathon 2020 (GOSSIS-derived)** ⭐ — 91,713 train + 39,308 test stays (~131k), 186 columns, target
   `hospital_death` (~8.6% positive); **openly downloadable via Kaggle** (Restricted licence on the PhysioNet mirror);
   multi-country; ships APACHE IVa death probabilities.
2. **MIMIC-IV v3.1** — 364,627 patients / 94,458 ICU stays; raw relational time series; **credentialed** (CITI+DUA); single US centre.
3. **MIMIC-III v1.4** — ~53k ICU stays; huge reproducible-benchmark literature; credentialed; older.
4. **eICU-CRD v2.0** — 200,859 stays across 208 units; multi-centre; credentialed; best for external-validity/fairness-by-site.

## Dataset Comparison

| Dataset | Records | Granularity | Multi-centre | Access | Best for |
|---|---|---|---|---|---|
| **WiDS 2020** | ~131k stays | First-24h aggregates | Yes (200+) | **Open (Kaggle)** | Fast start, benchmark vs APACHE |
| MIMIC-IV | 94k stays | Raw time series | No | Credentialed | Temporal, precise scores |
| MIMIC-III | ~53k stays | Raw time series | No | Credentialed | Reproducible benchmarks |
| eICU | ~200k stays | Semi-aggregated + APACHE | Yes (208) | Credentialed | External validity, fairness |

## Recommended Dataset

**Primary:** WiDS Datathon 2020 (via Kaggle) — build the full model→calibration→SHAP→API/dashboard on it.
**Advanced extension:** MIMIC-IV (raw time series, exact SOFA/SAPS reconstruction, temporal roadmap).
**External-validation stretch:** eICU (train on one, test across `hospital_id`).

## Why This Dataset Was Selected

WiDS is openly downloadable (no credentialing bottleneck), purpose-built for first-24h in-hospital mortality,
pre-windowed, realistically imbalanced, and ships **APACHE IVa probabilities** — a credible clinical baseline to
beat for free. That yields a fully reproducible, publicly shareable portfolio core, with a clean narrative up to
credentialed depth (MIMIC-IV) and multi-site validation (eICU). **Start CITI training early** — it gates the extensions.

## Data Dictionary

Identifiers `encounter_id`, `patient_id`, **`hospital_id`** (group for CV), `icu_id`. Demographics `age`, `gender`,
`ethnicity`, `bmi`. Context `icu_admit_source`, `icu_type`, `apache_2/3j_diagnosis`, `elective_surgery`,
`pre_icu_los_days`. First-24h vitals (`d1_*_min/max`: HR, MAP/SBP/DBP, resp rate, SpO₂, temp; `h1_` = first hour).
First-24h labs (`d1_*_min/max`: creatinine, BUN, sodium, potassium, glucose, bilirubin, albumin, haematocrit,
haemoglobin, WBC, platelets, lactate, pH, PaO₂/PaCO₂). Comorbidity flags (`aids`, `cirrhosis`, `hepatic_failure`,
`immunosuppression`, `leukemia`, `lymphoma`, `solid_tumor_with_metastasis`, `diabetes_mellitus`). APACHE covariates
+ **`apache_4a_hospital_death_prob`/`apache_4a_icu_death_prob`** (built-in baseline *and* leakage caution), GCS
components, `ventilated_apache`. Target `hospital_death`.

## Data Challenges

**Missingness is predominantly MNAR** — a lab is ordered *because* of concern, so being-measured is itself signal;
add missingness-indicator flags, never impute the fact away. **First-24h windowing / temporal leakage** — exclude
any end-of-stay field (discharge disposition, ICU LOS, discharge diagnosis, "expired"). **Outcome-model leakage** —
APACHE death-prob columns are outputs of another model (keep for baseline; train a variant without them). Class
imbalance (~8–9%). Non-independence (multiple stays per patient; clustering within hospitals) → grouped splits.
Impossible values/units → range-clip. Case-mix shift (GOSSIS ≠ NHS). Immortal-time bias for very-early deaths.

## Missing Values Strategy

Trees: pass NaNs through natively + explicit `*_measured` indicator flags (often top predictors). LogReg: median
imputation **inside folds** + indicators. **No SMOTE** (distorts calibration and manufactures implausible patients).

## Feature Engineering Ideas

First-24h min/max/mean (and first/last/std/count from raw MIMIC/eICU series) · deltas/ranges (`max−min`,
trajectory slopes) · clinical composites (shock index HR/SBP, pulse pressure, BUN:creatinine, anion gap,
PaO₂/FiO₂, reconstructed SOFA components, GCS total) · missingness indicators · comorbidity flags + Charlson/
Elixhauser burden · admission-context encodings. Fit all imputation/scaling inside CV folds. Keep APACHE prob for the baseline arm.

## Exploratory Data Analysis Plan

Target rate + per-hospital/ICU-type variation · missingness map + missingness-vs-outcome test (justifies MNAR
indicators) · vitals/labs distributions survivor vs non-survivor (flag implausible values) · single-feature AUROC
+ correlation with APACHE prob · redundancy clustering of min/max/mean · subgroup distributions (age/sex/ethnicity)
· manual column-by-column leakage audit · distributions by `hospital_id`.

## Machine Learning Strategy

Modelling ladder, each a stakeholder-legible comparison on identical folds: (1) clinical baseline APACHE IVa /
reconstructed SOFA & SAPS-II; (2) penalised LogReg ("recalibrated severity score"); (3) primary boosters
(XGBoost/LightGBM/CatBoost — native NaN routing, TreeSHAP). Grouped stratified CV (by `patient_id`; leave-hospitals-out
for external validity); nested CV for tuning; calibrate; choose operating points via decision-curve analysis and present risk tiers.

## Deep Learning Strategy (if applicable)

Only on raw time series (MIMIC-IV/eICU) as future work: LSTM/GRU/TCN/Transformer for **dynamic re-scoring** across
the first 24–48h rather than a single snapshot. Not needed for the WiDS aggregate core.

## Baseline Models

APACHE IVa probability (WiDS) / reconstructed SOFA & SAPS-II (MIMIC/eICU) as the incumbent clinical baseline, plus
penalised **Logistic Regression** as the transparent statistical floor.

## Advanced Models

**XGBoost, LightGBM, CatBoost** with `scale_pos_weight`; best model calibrated via **Platt or isotonic**
(`CalibratedClassifierCV`, held-out fold) — boosters are over-confident, so calibration materially improves Brier
and net benefit. Report reliability diagrams before/after.

## Model Comparison Plan

Grouped/leave-hospitals-out CV with bootstrap 95% CIs; head-to-head **ML vs APACHE IVa** (and SOFA/SAPS on
MIMIC/eICU) on the same folds; rank on AUPRC + calibration (Brier/ECE) with decision-curve utility; all logged to MLflow.

## Evaluation Metrics

**AUROC** (headline, literature-comparable) + **AUPRC** (honest under 8–9% prevalence) · **calibration** —
Brier, reliability curve, calibration slope/intercept, ECE (as important as discrimination for SMR-style audit) ·
**decision-curve / net benefit** · operating-point sens/spec/PPV/NPV/F1 · bootstrap CIs. Well-built boosters
commonly reach ~0.87–0.91 AUROC on WiDS-style data, but **lead with calibration and subgroup results**.

## Explainability Strategy

**TreeSHAP** (`shap.TreeExplainer`) — exact & fast. **Global:** beeswarm summary, dependence plots, mean|SHAP|
table for the model card. **Local (bedside):** SHAP waterfall + a plain-language top-drivers summary
("Risk 0.31 (High). Drivers: lactate 4.2, GCS 9, rising creatinine; mitigating: age 41, no metastatic disease").
Present **calibrated probability + tier + top 3–5 signed drivers**, not 180 raw values; state SHAP = association
within the model, not causation; keep dashboard and API explanations identical; design the UI to invite disagreement
and log overrides (guard against automation bias).

## Risk Assessment

Decision-support only — never an automated withdrawal/resource-denial trigger. Miscalibration is a direct harm in
prognostic conversations. **Label confounding:** in-hospital death is shaped by care decisions (e.g. treatment
withdrawal), so the model partly learns treatment-limitation patterns — document it. Expect calibration drift on NHS deployment.

## Ethical Considerations

Mandatory subgroup analysis (AUROC, AUPRC **and calibration**) by age/sex/ethnicity and hospital/site; note that
GOSSIS/MIMIC US ethnicity categories don't map cleanly to NHS categories. Automation-bias mitigation (override
logging, human-in-the-loop). No auto-actions.

## Data Privacy Notes

PhysioNet datasets are HIPAA de-identified; credentialed sets require **CITI + signed DUA** and **must never be
redistributed** (no raw CSVs in GitHub or Docker images; no PHI in outputs). Store only engineered de-identified
features + predictions/SHAP. Real NHS use → DPIA, Caldicott, DCB0129/0160 safety cases, MHRA SaMD consideration.

## Deployment Strategy

FastAPI (`/predict` + `/explain`) · Streamlit clinician dashboard · PostgreSQL (predictions, SHAP, clinician
feedback) · MLflow (tracking/registry) · Docker · GitHub Actions (CI incl. leakage + calibration gates) · drift monitoring.

## API Design

`POST /predict`
```json
{ "encounter_id":"A-10231","age":72,"gender":"F","bmi":24.1,"apache_2_diagnosis":113,
  "d1_heartrate_max":128,"d1_mbp_min":54,"d1_spo2_min":88,"d1_resprate_max":31,
  "d1_creatinine_max":2.4,"d1_lactate_max":4.2,"gcs_total":9,"ventilated_apache":1,
  "solid_tumor_with_metastasis":0 }
```
```json
{ "encounter_id":"A-10231","mortality_probability":0.312,"risk_tier":"HIGH",
  "model_version":"xgb-cal-isotonic-v1.4.2","threshold_used":0.15,"calibration":"isotonic",
  "disclaimer":"Decision support only. Not for automated treatment or resource decisions." }
```
`POST /explain` returns `base_value`, `predicted_probability`, and `top_contributions` (feature, value, signed SHAP, direction), `explainer:"TreeSHAP"`.

## Dashboard Ideas

Patient risk view (calibrated probability, tier, SHAP waterfall + plain-language drivers) · unit-level ranked
worklist · calibration/subgroup monitoring tab · prominent intended-use & uncertainty banners.

## Database Design

```sql
CREATE TABLE predictions (
  prediction_id BIGSERIAL PRIMARY KEY, encounter_id TEXT NOT NULL, model_version TEXT NOT NULL,
  mortality_prob NUMERIC(5,4) NOT NULL, risk_tier TEXT NOT NULL, threshold_used NUMERIC(5,4) NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now());
CREATE TABLE shap_contributions (
  id BIGSERIAL PRIMARY KEY, prediction_id BIGINT REFERENCES predictions,
  feature_name TEXT NOT NULL, feature_value DOUBLE PRECISION, shap_value DOUBLE PRECISION NOT NULL);
CREATE TABLE clinician_feedback (
  id BIGSERIAL PRIMARY KEY, prediction_id BIGINT REFERENCES predictions,
  agreed BOOLEAN, override_reason TEXT, created_at TIMESTAMPTZ DEFAULT now());
```
Store engineered de-identified features + predictions/SHAP only — never raw restricted source data.

## Folder Structure

```
03-icu-mortality/
├── src/icu_mortality/{config.py,data/,features/,models/,calibration/,explain/,api/,dashboard/}
├── tests/            # unit + leakage tests (window, outcome-model exclusion)
├── data/             # gitignored
├── notebooks/        # EDA only
├── docker/{Dockerfile,docker-compose.yml}
├── .github/workflows/ci.yml
├── Makefile ├─ pyproject.toml └─ README.md
```

## CI/CD Plan

GitHub Actions: lint (Ruff/Black) + MyPy + Pytest incl. **leakage tests** (assert only first-24h features; assert
APACHE-prob excluded in the "pure" model) and a **calibration gate** (fail if ECE/Brier regresses) → Docker build.

## Testing Strategy

Unit tests for feature aggregation (min/max/measured flags), calibration (monotonic reliability), SHAP additivity
(base + Σshap ≈ output), metrics; integration tests for `/predict` and `/explain` contracts; golden-file test on a pinned sample.

## MLOps Plan

MLflow logs params/AUROC/AUPRC/Brier/calibration artefacts; registry staged promotion; Docker Compose
(api+dashboard+db+mlflow); ship code + trained model, **not** data; model & data cards documenting intended use, subgroup performance, limitations.

## Monitoring Strategy

Track input **feature drift** (PSI/KS vs training), prediction drift, missingness-rate drift, and — when outcomes
arrive — rolling AUROC/AUPRC and **calibration** (the first thing to decay; alert if calibration slope departs from 1). Scheduled recalibration/retraining.

## Future Improvements

Temporal models on raw series (dynamic re-scoring) · external + prospective validation (WiDS/MIMIC → eICU → silent
NHS trial) · local recalibration to NHS/ICNARCH-2018-comparable outputs · competing-risk/survival framing ·
conformal prediction per patient · fairness-aware training + periodic re-audit · live EHR integration under governance.

## References

1. WiDS Datathon 2020 (Kaggle) — https://www.kaggle.com/c/widsdatathon2020 · data — https://www.kaggle.com/c/widsdatathon2020/data
2. WiDS 2020 (PhysioNet mirror) — https://physionet.org/content/widsdatathon2020/1.0.0/
3. MIMIC-IV v3.1 — https://physionet.org/content/mimiciv/3.1/
4. MIMIC-III v1.4 — https://physionet.org/content/mimiciii/1.4/
5. eICU-CRD v2.0 — https://physionet.org/content/eicu-crd/2.0/
6. Lundberg et al. 2020, Explainable AI for Trees (TreeSHAP), Nature Mach. Intell. — https://www.nature.com/articles/s42256-019-0138-9

## Useful Research Links

7. Lundberg & Lee 2017 (SHAP), NeurIPS — https://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions
8. TreeSHAP paper (arXiv:1802.03888) — https://arxiv.org/abs/1802.03888
9. Scoring systems in the critically ill (APACHE/SOFA/SAPS/ICNARC) — https://pmc.ncbi.nlm.nih.gov/articles/PMC7807847/
10. ICNARC (UK Case Mix Programme, ICNARCH-2018) — https://www.icnarc.org/
11. SHAP docs — https://shap.readthedocs.io/
12. scikit-learn calibration — https://scikit-learn.org/stable/modules/calibration.html
13. XGBoost — https://xgboost.readthedocs.io/ · LightGBM — https://lightgbm.readthedocs.io/ · CatBoost — https://catboost.ai/
14. Papers With Code — ICU mortality prediction — https://paperswithcode.com/task/mortality-prediction
15. MIT-LCP mimic-code — https://github.com/MIT-LCP/mimic-code
16. Harutyunyan et al. MIMIC-III benchmarks — https://github.com/YerevaNN/mimic3-benchmarks

---

*Access nuance: WiDS is genuinely open via Kaggle; MIMIC-III/IV and eICU are credentialed (CITI + DUA) and never
redistributable. Lead with calibration + subgroup fairness and an explicit head-to-head against APACHE IVa.*
