# 01 — Hospital Readmission Prediction System (30-Day)

> **Design document.** Predict, at or before discharge, the probability that a patient is readmitted as an
> emergency within **30 days** — a calibrated, explainable decision-support tool to target scarce discharge
> and community resources. Trained on an **open** patient-level dataset, framed with **NHS** definitions.

---

## Executive Summary

Unplanned 30-day readmissions are a national NHS quality indicator (NHS Outcomes Framework 3b) and have
carried direct financial consequences for providers since the 2011/12 Payment-by-Results non-payment rule.
This system ranks patients by readmission risk so discharge teams, pharmacists, and virtual-ward services can
intervene where it matters. The build uses the **UCI Diabetes 130-US Hospitals** dataset (CC BY 4.0, 101,766
encounters) because it is the only option that is openly redistributable, letting reviewers clone-and-run —
with **MIMIC-IV** documented as the credentialed "phase 2" extension. The emphasis is on the things that make
readmission models fail in practice: leakage (death/hospice rows, patient-level grouping), calibration, and
net-benefit evaluation rather than accuracy.

## Problem Statement

Given features available at discharge, output a calibrated probability of unplanned readmission within 30 days
(binary classification). Success = **beating a sensible baseline on AUPRC + calibration**, with a decision
threshold chosen from capacity/cost, evaluated by decision-curve analysis. Realistic AUROC on this dataset is
~0.64–0.70; anything ≳0.9 signals leakage.

## NHS Relevance

NHS England tracks **emergency readmissions within 30 days of discharge** (NHSOF 3b; CCG/ICB Outcomes Indicator
3.2). Under Payment by Results (from 2011/12), commissioners applied a **non-payment / marginal-rate rule** for
excess 30-day emergency readmissions, redirecting withheld funds to post-discharge support — a direct incentive
to reduce avoidable readmissions. The US HRRP (CMS) penalises the same metric, which is why the problem is so
well-studied and why public data exists.

## Business Value

Reduced avoidable readmissions → freed bed-days, lower tariff-penalty exposure, better patient experience, and
cheaper *targeted* community intervention versus blanket programmes. The economic case rests on the marginal
cost of a readmission versus the cost of intervening on true positives — hence net-benefit analysis is central.

## Expected Users

Discharge/transfer-of-care teams and virtual-ward services who need a prioritised worklist at discharge, plus
pharmacists (medicines optimisation) and operational leads (aggregate risk trends).

## Stakeholders

| Stakeholder | Decision supported |
|---|---|
| Discharge / transfer-of-care teams | Which patients get an enhanced discharge bundle + early follow-up |
| Virtual ward / hospital-at-home | Prioritised proactive follow-up within 48–72h |
| Clinicians (consultants, GPs) | Adjust discharge timing, specialist follow-up, polypharmacy review |
| Pharmacists | Targeted discharge counselling / medicines reconciliation |
| Operational & finance leads | Capacity planning; reduce PbR penalty exposure |
| ICB / population-health analysts | Commission community/re-ablement services |
| Governance / Caldicott Guardian | Assurance, fairness, IG sign-off |

## Dataset Options

1. **UCI Diabetes 130-US Hospitals (1999–2008)** ⭐ — 101,766 encounters, ~47 modelling features, **CC BY 4.0**, target `readmitted` ∈ {`<30`,`>30`,`NO`}; diabetic inpatients, US, tabular.
2. **MIMIC-IV v3.1** — 364,627 patients / ~546k admissions; relational EHR; **PhysioNet credentialed** (CITI + DUA); readmission label must be engineered.
3. **eICU-CRD v2.0** — 200,859 ICU stays across 208 hospitals; multi-centre; credentialed; ICU-only.
4. **NHS HES/SUS (patient-level)** — directly NHS but **DARS-restricted**, not usable for an open portfolio. NHS publishes only **aggregate** 30-day readmission stats (context/benchmark).

## Dataset Comparison

| Dataset | Records | Level | Access | Target ready | NHS-relevant | Use |
|---|---|---|---|---|---|---|
| **UCI Diabetes 130** | 101,766 | Encounter | **Open (CC BY 4.0)** | Yes | Analogous (US) | **Primary** |
| MIMIC-IV | ~546k adm | Relational | Credentialed | Engineer | Rich, US ICU-centric | Phase 2 |
| eICU | ~200k stays | Relational ICU | Credentialed | Engineer | Multi-centre | Fairness extension |
| HES/SUS | National | Episode | DARS (closed) | Yes | **Direct** | Not feasible |

## Recommended Dataset

**UCI Diabetes 130-US Hospitals** as the primary training set, problem **framed and evaluated with NHS 30-day
emergency-readmission definitions**, and **MIMIC-IV** documented as the credentialed extension.

## Why This Dataset Was Selected

(1) **CC BY 4.0** — the only redistributable option, so the repo is clone-and-run for reviewers; (2) no
credentialing barrier → fastest reproducible end-to-end system; (3) a well-understood label with a rich
benchmark literature and known pitfalls to demonstrate seniority; (4) tabular → clean showcase of the full
production stack; (5) MIMIC-IV as the credible next step signals awareness of teaching-data vs production reality.

## Data Dictionary

Key fields: `encounter_id`, **`patient_nbr`** (patient id — critical for grouping), `race`, `gender`, `age`
(10-yr bands), `admission_type_id`, `discharge_disposition_id`, `admission_source_id`, `time_in_hospital`,
`num_lab_procedures`, `num_procedures`, `num_medications`, `number_outpatient/emergency/inpatient` (prior-year
utilisation — strongest predictors), `diag_1/2/3` (ICD-9), `number_diagnoses`, `max_glu_serum`, `A1Cresult`,
23 drug columns (e.g. `insulin` ∈ {No,Steady,Up,Down}), `change`, `diabetesMed`, target `readmitted`.

## Data Challenges

**Leakage traps (where seniority shows):** (1) **death/hospice discharge dispositions** (codes 11,13,14,19,20,21)
describe patients who cannot be readmitted — **drop these rows** (Strack et al.); (2) a `patient_nbr` recurs
across encounters — random splits leak patients → use **grouped CV**; (3) collapsing `>30` into positive answers
the wrong question (only `<30` is positive); (4) high-cardinality ICD-9 diag codes (~700+) must be grouped;
(5) high missingness in `weight` (~97%), `medical_specialty`/`payer_code` (~40–50%).

## Missing Values Strategy

`weight` → drop (optional "recorded" flag); `medical_specialty`/`payer_code`/`race` → explicit **"Unknown"
category** (missingness is MNAR and informative); `diag_2/3` sparse → grouped category incl. "None"; convert
`?`→NaN first; never median-impute categorical-looking numeric IDs.

## Feature Engineering Ideas

Prior-utilisation intensity (sum & ratios of prior visits; "any prior inpatient") · comorbidity burden
(`number_diagnoses`; map `diag_1/2/3` to Charlson/Elixhauser or Strack's ~9 groups) · medication complexity
(active-drug count, polypharmacy flag, insulin status) · glycaemic control (`A1Cresult`, `max_glu_serum`;
A1C-measured × primary-diagnosis interaction) · service intensity (LOS, labs, procedures/day) · grouped
admission context (after removing death/hospice) · age midpoint. Encode model-appropriately (one-hot for LogReg;
native categoricals for LightGBM/CatBoost).

## Exploratory Data Analysis Plan

Integrity (missing map, encounters-per-patient) · target balance *after* removing death/hospice (quantify rows
dropped) · univariate distributions · readmission-rate stratified by prior inpatient visits, `number_diagnoses`,
A1C-measured, insulin, primary-diagnosis group, discharge disposition · diag-code cardinality → grouping design ·
leakage checks · MNAR test (readmission rate for missing vs present) · fairness slices by race/gender/age.

## Machine Learning Strategy

**Validation defines everything:** StratifiedGroupKFold (5-fold) grouped on `patient_nbr`, plus an untouched
grouped test set; all preprocessing inside a `Pipeline`/`ColumnTransformer` fit per fold. Progression: LogReg
baseline → gradient boosting → calibration → cost/capacity threshold. Selection on **AUPRC + calibration**, not AUROC/accuracy.

## Deep Learning Strategy (if applicable)

Not required for this tabular problem — gradient-boosted trees dominate. DL is only relevant at the MIMIC-IV
extension (sequence models over prior encounters); documented as future work.

## Baseline Models

**Logistic Regression** (`class_weight='balanced'`, L1/L2) with one-hot + scaling — interpretable, reports
odds ratios, sets the bar. Random Forest as a quick sanity check.

## Advanced Models

**XGBoost, LightGBM (native categoricals), CatBoost**, tuned with **Optuna** on grouped CV; `scale_pos_weight`
for imbalance; best model wrapped in **`CalibratedClassifierCV`** (isotonic/Platt) on a separate calibration fold.

## Model Comparison Plan

Grouped CV with bootstrap 95% CIs; compare LogReg vs boosters on AUPRC/AUROC/Brier/ECE and **net benefit** at
the operating point; log all runs to MLflow; pick on AUPRC + calibration, tie-break on net benefit.

## Evaluation Metrics

**AUPRC** (primary; baseline = prevalence ~0.11) · AUROC (report, note prevalence-insensitivity) · **calibration**
(reliability curve, Brier, ECE — non-negotiable) · **decision-curve / net-benefit** (Vickers & Elkin) · operating-point
precision/recall/specificity/PPV/NPV/F-beta · subgroup metrics. **Accuracy excluded as a headline** (~89% trivial).

## Explainability Strategy

Global **SHAP** (TreeExplainer beeswarm + dependence) cross-checked with permutation importance; **local SHAP
waterfall** surfaced per patient in UI/API ("3 prior inpatient admissions; 18 medications; discharged home without
home-health"); LogReg odds ratios as a transparent reference. Explanations support clinician challenge, not automation.

## Risk Assessment

Decision-support only — a clinician owns the discharge decision; guard against automation bias. **Distribution
shift**: 1999–2008 US diabetic inpatients ≠ live NHS population — no deployment without local recalibration and
validation. State this loudly.

## Ethical Considerations

Audit performance/error by **race, gender, age** (Fairlearn ships this dataset for fairness teaching): subgroup
AUROC/AUPRC, calibration-by-group, equalized-odds gaps, flag-rate parity. Watch the **allocation feedback loop**
(resources following scores can entrench under-service of low-utilisation groups).

## Data Privacy Notes

UCI data is already de-identified, but treat the pipeline as if handling PID (pseudonymise, encrypt, RBAC, audit
logs, retention policy). Any NHS/MIMIC extension → **UK GDPR + DPA 2018**, Caldicott, **DSPT**, DPIA, and an
Algorithmic Transparency Recording Standard entry; MIMIC/eICU require credentialing + DUA and **must not be
committed to the repo**.

## Deployment Strategy

FastAPI (serving) · Streamlit (dashboard) · PostgreSQL (encounters, predictions, outcomes) · MLflow
(tracking/registry) · Docker Compose · GitHub Actions (CI/CD) · Evidently (drift).

## API Design

`POST /predict`
```json
{ "age":"[70-80)","gender":"Female","race":"Caucasian","time_in_hospital":6,"num_medications":18,
  "number_diagnoses":9,"number_inpatient":2,"number_emergency":1,"number_outpatient":0,
  "admission_type":"Emergency","discharge_disposition":"Discharged to home","A1Cresult":">8",
  "insulin":"Up","diabetesMed":"Yes","change":"Ch","primary_diagnosis_group":"Circulatory" }
```
```json
{ "readmission_probability":0.31,"risk_band":"High","threshold_used":0.18,"flag":true,
  "model_version":"lightgbm_v1.4.2",
  "top_factors":[{"feature":"number_inpatient","value":2,"shap":0.44},
                 {"feature":"num_medications","value":18,"shap":0.21}],
  "disclaimer":"Decision-support only; not a substitute for clinical judgement." }
```
Also `GET /health`, `GET /model/info`, `POST /predict/batch`. Pydantic validates categorical domains & ranges.

## Dashboard Ideas

Single-patient scoring form + risk gauge + SHAP waterfall · ward/cohort worklist (sortable by risk, filterable) ·
monitoring tab (rolling AUROC/AUPRC, calibration drift, feature drift, fairness slices) · threshold explorer
(precision/recall/net-benefit vs capacity).

## Database Design

```sql
CREATE TABLE patients (patient_id BIGINT PRIMARY KEY, age_band TEXT, gender TEXT, race TEXT);
CREATE TABLE encounters (
  encounter_id BIGINT PRIMARY KEY, patient_id BIGINT REFERENCES patients,
  admit_time TIMESTAMP, discharge_time TIMESTAMP, discharge_disp TEXT,
  time_in_hospital INT, features_json JSONB);
CREATE TABLE predictions (
  prediction_id BIGSERIAL PRIMARY KEY, encounter_id BIGINT REFERENCES encounters,
  model_version TEXT, probability NUMERIC(5,4), threshold_used NUMERIC(5,4),
  flag BOOLEAN, top_factors JSONB, created_at TIMESTAMP DEFAULT now());
CREATE TABLE outcomes (
  encounter_id BIGINT REFERENCES encounters, readmitted_30d BOOLEAN, observed_at TIMESTAMP);
```
`predictions` + `outcomes` enable live performance/drift monitoring and audit.

## Folder Structure

```
01-hospital-readmission/
├── src/readmission/{config.py,data/,features/,models/,evaluate/,api/,dashboard/}
├── tests/            # unit + leakage tests (death/hospice removal, group split)
├── data/             # gitignored
├── notebooks/        # EDA only
├── docker/{Dockerfile,docker-compose.yml}
├── .github/workflows/ci.yml
├── Makefile
├── pyproject.toml
└── README.md
```

## CI/CD Plan

GitHub Actions: Ruff + Black --check + MyPy + Pytest (incl. **leakage tests**: assert death/hospice rows removed,
assert no `patient_nbr` crosses train/test) → train/eval smoke test with an **AUPRC gate** → Docker build.

## Testing Strategy

Unit tests for preprocessing (row removal, encoding, Unknown-category), features (no leakage), metrics; a
**leakage test suite**; integration tests for API (FastAPI TestClient) and a golden-file test on a pinned sample.

## MLOps Plan

MLflow logs params/CV metrics/calibration+PR curves/SHAP/feature-list/data-hash; registry Staging→Production;
Docker Compose (api + streamlit + postgres + mlflow); reproducible pipelines via config + seeds + data hash.

## Monitoring Strategy

Evidently data/prediction-drift reports on a schedule; track **calibration drift** and **subgroup performance**;
once `outcomes` accrue, rolling realised precision/recall → retraining trigger; monitor flag-rate equity.

## Future Improvements

MIMIC-IV/eICU for labs/vitals/notes + multi-centre external validation · temporal/survival framing (competing
risk of death) · condition-specific models (HF/COPD/pneumonia, HRRP-style) · **uplift/causal** modelling (who
benefits from intervention) · live recalibration on NHS data · silent prospective trial + health-economic
evaluation · federated learning across trusts · Fairlearn-constrained training if gaps appear.

## References

1. UCI Diabetes 130-US Hospitals — https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008
2. OpenML mirror (id 4541) — https://www.openml.org/d/4541
3. Strack et al. 2014 (origin paper) — https://onlinelibrary.wiley.com/doi/10.1155/2014/781670
4. MIMIC-IV v3.1 — https://physionet.org/content/mimiciv/3.1/
5. eICU-CRD v2.0 — https://physionet.org/content/eicu-crd/2.0/
6. Vickers & Elkin 2006 (Decision Curve Analysis) — https://pubmed.ncbi.nlm.nih.gov/17099194/

## Useful Research Links

7. NHS Digital — Emergency readmissions within 30 days (compendium) — https://digital.nhs.uk/data-and-information/publications/statistical/compendium-emergency-readmissions/current
8. NHS indicator 3.2 specification — https://digital.nhs.uk/data-and-information/publications/statistical/ccg-outcomes-indicator-set/specifications/3.2-emergency-readmissions-within-30-days-of-discharge-from-hospital_1_4
9. data.gov.uk NHSOF 3b — https://www.data.gov.uk/dataset/52c86bb2-d524-4fbf-aaa5-3b336814b767/emergency-readmissions-within-30-days-of-discharge-from-hospital-nhsof-3b
10. Nuffield Trust — Emergency readmissions — https://www.nuffieldtrust.org.uk/resource/emergency-readmissions
11. CMS HRRP — https://www.cms.gov/medicare/quality/value-based-programs/hospital-readmissions
12. Fairlearn — Diabetes-130 fairness guide — https://fairlearn.org/main/user_guide/datasets/diabetes_hospital_data.html
13. scikit-learn — Probability calibration — https://scikit-learn.org/stable/modules/calibration.html
14. scikit-learn — StratifiedGroupKFold — https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html
15. SHAP — https://shap.readthedocs.io/
16. Evidently AI — https://www.evidentlyai.com/ · MLflow — https://mlflow.org/

---

*Realistic target AUROC ~0.65–0.70; lead with AUPRC + calibration + net benefit. Signature pitfalls to
demonstrate mastery of: death/hospice removal, patient-level grouped CV, calibration, and honest NHS-vs-US
population framing.*
