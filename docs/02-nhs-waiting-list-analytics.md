# 02 — NHS Waiting List Analytics & Forecasting System

> **Design document.** Analyse NHS waiting-list dynamics, surface trends and bottlenecks by trust and
> specialty, and forecast future demand/backlog with calibrated uncertainty — built on **real NHS England
> open data** (Open Government Licence v3.0). No patient-identifiable data is used.

---

## Executive Summary

The NHS publishes monthly, trust-level time series on how long patients wait for elective care,
emergency care, cancer treatment, and diagnostics. Waiting times are the single most politically and
operationally scrutinised NHS KPI, and the elective backlog sits at historic highs (~7.3m RTT pathways).
This system turns those static monthly spreadsheets into **reproducible trend analysis + probabilistic
forecasts**, so trusts and planners can intervene *earlier* and *where the bottleneck actually is*.

The technical core is a monthly forecasting engine for the **RTT incomplete waiting list** (national and
per-specialty), benchmarked against a seasonal-naive baseline, with explicit handling of the COVID-19
structural break, rolling-origin backtesting, and MASE-led evaluation. Because the data is aggregate and
open, the project is low-risk to build, fully reproducible, and shareable.

## Problem Statement

Given monthly historical waiting-list data at provider × specialty × waiting-time-band granularity,
**(a)** characterise trend, seasonality and structural breaks; **(b)** identify the trusts/specialties
driving the backlog; and **(c)** forecast the incomplete-pathway list 6–12 months ahead with calibrated
prediction intervals. Success = forecasts that **beat seasonal-naive on MASE** on out-of-sample rolling
origins, with 95% intervals whose empirical coverage is ≈95%.

## NHS Relevance

Four constitutional/operational standards frame the domain:

| Standard | Metric | Target |
|---|---|---|
| **RTT** | % of *incomplete* pathways waiting ≤ 18 weeks | ≥ 92% (unmet nationally since ~2016) |
| **A&E 4-hour** | % dealt with within 4 hours | 95% constitutional; 78% interim recovery |
| **Cancer 62-day** | urgent referral → first treatment | ≥ 85% |
| **Cancer 28-day FDS** | referral → diagnosis/rule-out | ≥ 75% |
| **Diagnostics (DM01)** | % waiting ≥ 6 weeks | < 1% |

The same open data underpins analyses by the BMA, Nuffield Trust, King's Fund and the House of Commons
Library — this project reproduces and extends that work with forecasting.

## Business Value

Replace lagging monthly spreadsheets with forward-looking, uncertainty-aware planning intelligence:
earlier detection of deteriorating specialties, capacity-targeted intervention, and scenario testing
(e.g. "what does +10% monthly completions do to the 52-week waiters?").

## Expected Users

Operational analysts and planners who currently work from the raw NHS workbooks, plus dashboard
consumers (managers, commissioners) who need filtered, visual, forecast-aware views.

## Stakeholders

| Stakeholder | Decision supported |
|---|---|
| Trust operational managers / COOs | Where are breaches accumulating; which specialties need extra capacity |
| Integrated Care Boards (42 ICBs) | Allocate elective/diagnostic capacity across member trusts |
| NHS England Elective Recovery team | Track vs recovery trajectories; forecast national list; scenario-test |
| Regional assurance teams | Benchmark trusts; spot deteriorating outliers early |
| DHSC / policy | Funding cases, mandate targets, public accountability |
| Public / researchers | Transparency (mirrors BMA / Nuffield Trust analyses) |

## Dataset Options

All published by **NHS England**, monthly, provider + ICB level, **OGL v3.0**.

1. **Consultant-led RTT Waiting Times** ⭐ — monthly since Mar 2007 (current format Oct 2015); provider × treatment-function-code × waiting-time band; **zipped full CSV** available; the headline "waiting list".
2. **A&E Attendances & Emergency Admissions** — monthly; Type 1/2/3 attendances, 4-hour performance, trolley waits; **Excel only** (multi-tab); strong winter seasonality.
3. **Cancer Waiting Times (CWT)** — monthly; provider/ICB × standard × tumour site; CSV + Excel; standards reformed Oct 2023 (breaks comparability).
4. **Diagnostics (DM01)** — monthly since **Jan 2006** (longest, cleanest); 15 tests; CSV + Excel; excellent *leading indicator* of elective demand.

## Dataset Comparison

| Criterion | RTT | A&E | Cancer | DM01 |
|---|---|---|---|---|
| History length | 2007+ | 2010+ | 2009/2022+ | **2006+ (longest)** |
| Machine-readable | Zipped CSV ✅ | Excel only ⚠️ | CSV+Excel ✅ | CSV+Excel ✅ |
| Dimension richness | **Very high** | Low | Medium | Medium |
| Forecasting signal | Strong trend + break | Strong seasonality | Noisy + reform break | Strong, clean |
| Policy salience | **Highest** | High | High | Medium-high |

## Recommended Dataset

**Primary:** RTT **incomplete-pathway** waiting list (national → per-specialty).
**Secondary:** DM01 (clean long series + leading-indicator feature for RTT demand).
**Complementary dashboard series:** A&E (seasonality showcase) and Cancer (breach-rate/ICB views).

## Why This Dataset Was Selected

RTT *is* the "NHS waiting list": highest policy salience, richest specialty granularity for bottleneck
analysis, and — unlike A&E — it ships as machine-readable zipped CSV. DM01 adds a long, low-noise series
whose 6-week breaches lead future RTT demand, giving a genuine exogenous predictor rather than a toy feature.

## Data Dictionary

**RTT Part (pathway) types:** *Incomplete* (still waiting at month-end — **the forecasting target**),
*Incomplete with DTA*, *Completed Admitted*, *Completed Non-Admitted*, *New RTT Periods* (demand proxy).
**Waiting-time bands:** weekly `>0–1 … >51–52`, then `>52` (incomplete extends to `>104`), plus total and
"unknown clock start". **18-week breaches** = Σ bands `>18`; 52/65/78-week waiters derived similarly.
**Treatment Function Codes (selected):** 110 Trauma & Orthopaedics, 130 Ophthalmology (largest waiters),
100 General Surgery, 120 ENT, 320 Cardiology, 330 Dermatology, 502 Gynaecology, **C_999 = all specialties**.

## Data Challenges

Format drift (RTT format change Oct 2015; incomplete cap 52→104w) · **COVID structural break (Mar–Apr 2020)**
· seasonality · missing/late months & non-submitting trusts · provider mergers (org-code churn) · revisions
(provisional vs final vintages) · small-denominator noise for rare specialties/tumours.

## Missing Values Strategy

Explicit handling, never silent zero-fill: flag non-submitting trust-months; interpolate only short internal
gaps with a missingness indicator; maintain an **org-code lineage map** so mergers don't create phantom level
shifts; keep every **data vintage** so revisions are auditable.

## Feature Engineering Ideas

Calendar (month, quarter, cyclical month, working-days-in-month, financial-year period, winter flag) · target
lags (t−1,2,3,12) and cross-series lags (DM01 6-week breaches, New RTT periods) · rolling mean/std/min/max
(3/6/12m), MoM & YoY deltas · backlog metrics (clearance rate, net list change, %>18/52/65/78w, interpolated
median/92nd-percentile wait) · intervention flags (COVID dummy, post-2015 format, Oct-2023 cancer reform) ·
per-capita normalisation via **ONS mid-year population**.

## Exploratory Data Analysis Plan

Trend (indexed to Jan-2019) · STL seasonal decomposition + seasonal-amplitude quantification ·
**change-point detection** (`ruptures`/CUSUM/Chow) to locate COVID and other breaks empirically · breach-rate
& long-waiter trends · trust/ICB choropleths & rankings · specialty Pareto (which TFCs drive the backlog) ·
cross-correlation (DM01 → RTT lead/lag) · data-quality audit (missing months, org churn, revision magnitude).

## Machine Learning Strategy

Global vs per-series models, always benchmarked against the baseline before adding complexity (see model ladder).

## Deep Learning Strategy (if applicable)

Optional and only after simpler models are beaten on backtest: global **LSTM / Temporal Fusion Transformer /
N-BEATS / N-HiTS** trained across trusts & specialties via `darts` / `pytorch-forecasting`.

## Baseline Models

**Naive** (last value), **Seasonal-naive** (value 12 months ago), **drift**. Mandatory reference — a
peer-reviewed 2026 audit found that on COVID-inclusive windows *no* modern ML method reliably beat
seasonal-naive, so it is the bar every model must clear.

## Advanced Models

**SARIMA/SARIMAX** (statsmodels; inject COVID dummies + exogenous DM01/referrals) · **ETS/Holt-Winters** ·
**Prophet** (changepoints + seasonality + COVID regressor; interpretable decomposition) · **LightGBM/XGBoost
on engineered lags** (global multi-series, direct or recursive multi-horizon).

## Model Comparison Plan

Rolling-origin (expanding-window) backtest with a fixed horizon, refitting at each origin across pre-COVID,
COVID and recovery periods; **Diebold–Mariano** significance tests vs seasonal-naive; log every run to MLflow
and rank by MASE + interval coverage.

## Evaluation Metrics

**MASE** (primary — scale-free, MASE<1 beats naive) · MAE / RMSE (patients) · MAPE/sMAPE (with care on small
series) · **prediction-interval quality**: empirical 80/95% coverage, interval width, pinball/quantile loss, CRPS.

## Explainability Strategy

Prophet/STL trend + seasonal + changepoint plots ("why the list is moving") · SHAP / gain importance on the
lag-feature GBM (e.g. DM01 breaches & referral lags driving the forecast) · **scenario what-ifs** (completions
uplift, referral surge) exposed as dashboard sliders · list-growth attribution (starts vs completions; per
specialty/trust).

## Risk Assessment

Primary risk is **misinterpretation**, not privacy: forecasts used to performance-manage trusts, or presented
without uncertainty. Mitigate with mandatory intervals, documented COVID break, provisional/final vintage
labels, and no causal claims from correlational features.

## Ethical Considerations

Avoid naming-and-shaming framing; present per-capita and deprivation-aware views; note that waiting-list data
reflects *recorded* activity, not unmet need; respect NHS small-number suppression (don't re-derive disclosive
granular ratios).

## Data Privacy Notes

**No patient-level PII** — all four datasets are aggregate trust/specialty counts, so there are no special-category
GDPR records. Cite OGL v3.0 attribution; version data vintages; log model assumptions.

## Deployment Strategy

FastAPI (serving) · Streamlit (dashboard) · PostgreSQL (store, vintage-aware) · MLflow (tracking/registry) ·
Docker Compose · GitHub Actions (CI + scheduled monthly refresh after NHS publication day).

## API Design

`POST /forecast`
```json
{ "dataset": "rtt_incomplete", "provider_code": "RJ1", "treatment_function": "110",
  "horizon_months": 6, "model": "sarimax", "interval": 0.95 }
```
```json
{ "series_id": "rtt_incomplete:RJ1:110", "model": "sarimax", "origin": "2026-05",
  "forecast": [ {"period":"2026-06","yhat":4231,"lower":3980,"upper":4487},
                {"period":"2026-07","yhat":4310,"lower":4001,"upper":4629} ],
  "metrics_backtest": {"MASE":0.83,"MAE":142,"coverage_95":0.94},
  "data_vintage": "2026-05-rtt-v1" }
```
Other endpoints: `GET /series/{id}`, `GET /breaches`, `POST /scenario`, `GET /models`, `GET /health`.

## Dashboard Ideas

Filters (trust, ICB/region, TFC, pathway type, date range) · **breach heatmap** (trust × specialty, %>18w) ·
forecast **fan chart** (80/95% bands, model toggle) · trend + seasonal decomposition panel · 52w+ long-waiter
tracker · scenario sliders → live re-forecast · per-capita geographic choropleth.

## Database Design

```sql
CREATE TABLE dim_provider (provider_code TEXT PRIMARY KEY, name TEXT, icb_code TEXT, region TEXT);
CREATE TABLE dim_specialty (tfc TEXT PRIMARY KEY, name TEXT);
CREATE TABLE fact_waiting (
  dataset TEXT, period DATE, provider_code TEXT REFERENCES dim_provider,
  tfc TEXT REFERENCES dim_specialty, pathway_type TEXT,
  total INTEGER, over_18wk INTEGER, over_52wk INTEGER,
  data_vintage TEXT, loaded_at TIMESTAMPTZ DEFAULT now(),
  PRIMARY KEY (dataset, period, provider_code, tfc, pathway_type, data_vintage));
CREATE TABLE fact_forecast (
  series_id TEXT, model TEXT, origin DATE, period DATE,
  yhat DOUBLE PRECISION, lower DOUBLE PRECISION, upper DOUBLE PRECISION,
  run_id TEXT, PRIMARY KEY (series_id, model, origin, period));
```
`data_vintage` in the PK preserves every revision — essential for leak-free backtesting.

## Folder Structure

```
02-waiting-list-analytics/
├── src/waiting_list/
│   ├── config.py            # Pydantic Settings
│   ├── ingest/              # format-versioned NHS file loaders (CSV/Excel)
│   ├── features/            # calendar, lags, backlog metrics
│   ├── models/              # baselines, sarimax, prophet, gbm, (dl)
│   ├── backtest/            # rolling-origin evaluation
│   ├── api/                 # FastAPI app
│   └── dashboard/           # Streamlit app
├── tests/                   # pytest (unit + integration)
├── data/                    # gitignored (raw/interim/processed)
├── notebooks/               # EDA only
├── docker/  ├─ Dockerfile  └─ docker-compose.yml
├── .github/workflows/       # ci.yml, monthly-refresh.yml
├── Makefile
├── pyproject.toml
└── README.md
```

## CI/CD Plan

GitHub Actions: **ci.yml** (Ruff + Black --check + MyPy + Pytest + Docker build on push/PR) and
**monthly-refresh.yml** (cron after NHS publication: fetch → schema-validate → ingest → retrain → log drift).

## Testing Strategy

Unit tests for ingestion (format versions, org-code mapping), feature builders (no future leakage), and metric
functions (MASE against known cases); integration tests for the backtest harness and API endpoints (FastAPI
TestClient); a golden-file test on a small pinned data sample.

## MLOps Plan

MLflow logs params/metrics/artifacts per model×series; registry promotes best model per series; Docker Compose
runs api+dashboard+db+mlflow; scheduled retrain workflow; reproducible pipelines via config + pinned data vintages.

## Monitoring Strategy

Data-freshness checks, schema/format-drift alerts, rolling backtest **MASE vs seasonal-naive** (alert/retrain on
degradation), and **interval-coverage** monitoring (flag if 95% band coverage drifts from nominal).

## Future Improvements

Hierarchical/reconciled forecasting (trust→ICB→national) · global deep models (TFT/N-HiTS) · conformal
prediction for guaranteed coverage · exogenous drivers (bed occupancy, CDC capacity, flu) · automated
change-point re-detection · what-if capacity optimisation linked to a queue/simulation model.

## References

1. RTT Waiting Times — https://www.england.nhs.uk/statistics/statistical-work-areas/rtt-waiting-times/
2. RTT 2025-26 data files — https://www.england.nhs.uk/statistics/statistical-work-areas/rtt-waiting-times/rtt-data-2025-26/
3. RTT collection/metadata — https://digital.nhs.uk/data-and-information/data-collections-and-data-sets/data-collections/consultant-led-referral-to-treatment-waiting-times-rtt
4. A&E Attendances & Emergency Admissions — https://www.england.nhs.uk/statistics/statistical-work-areas/ae-waiting-times-and-activity/
5. Cancer Waiting Times — https://www.england.nhs.uk/statistics/statistical-work-areas/cancer-waiting-times/
6. Diagnostics DM01 — https://www.england.nhs.uk/statistics/statistical-work-areas/diagnostics-waiting-times-and-activity/monthly-diagnostics-waiting-times-and-activity/

## Useful Research Links

7. ONS mid-year population estimates — https://www.ons.gov.uk/peoplepopulationandcommunity/populationandmigration/populationestimates/datasets/estimatesofthepopulationforenglandandwales
8. statsmodels SARIMAX — https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html
9. Prophet — https://facebook.github.io/prophet/
10. Nixtla StatsForecast (fast baselines) — https://github.com/Nixtla/statsforecast
11. Darts (classical + deep forecasting) — https://github.com/unit8co/darts
12. Forecasting: Principles & Practice (MASE, rolling-origin) — https://otexts.com/fpp3/
13. Forecasting NHS demand under two structural breaks (2026) — https://www.sciencedirect.com/science/article/abs/pii/S1386505626003199
14. Modelling elective waiting-list recovery post-COVID (PMC) — https://pmc.ncbi.nlm.nih.gov/articles/PMC9365524/
15. BMA NHS backlog analysis — https://www.bma.org.uk/advice-and-support/nhs-delivery-and-workforce/pressures/nhs-backlog-data-analysis
16. Open Government Licence v3.0 — https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/

---

*Data URLs verified live (July 2026). Build order within project: ingestion → EDA → baseline → SARIMAX/Prophet →
GBM → backtest harness → API/dashboard → MLOps. Seasonal-naive is the bar; COVID break handled explicitly;
evaluation is MASE-led with calibrated intervals.*
