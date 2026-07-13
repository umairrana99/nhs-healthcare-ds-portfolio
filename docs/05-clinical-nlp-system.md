# 05 — Clinical NLP Information Extraction System

> **Design document.** Extract medical entities — diseases, medications, symptoms, procedures, tests, anatomy,
> and medication signatures (dose/frequency/route) — from free-text clinical notes using transformer-based NER,
> with optional normalization to SNOMED CT / UMLS / RxNorm / ICD-10. Built on **openly licensed** corpora; PHI/DUA
> handling is the central design constraint.

---

## Executive Summary

Clinical narrative (discharge summaries, progress notes, referral letters) holds detail that the structured coded
EHR loses. This system does two things: transformer **clinical NER** (span + type) and optional **entity linking**
to controlled terminologies — directly supporting NHS clinical coding, cohort discovery, and pharmacovigilance.
The public portfolio build uses **openly downloadable** corpora (BC5CDR + NCBI-Disease + MACCROBAT) with
**BiomedBERT / Bio_ClinicalBERT** token-classifiers, **medspaCy** negation/assertion, and **scispaCy** UMLS
linking — evaluated **entity-level with seqeval**. MIMIC/n2c2/i2b2 are documented as the credentialed, never-redistributed
production path. The hard constraint throughout: **PHI stays inside the trust boundary; restricted data and any
PHI-bearing outputs are never committed or sent to third-party APIs.**

## Problem Statement

(1) Clinical NER: locate and type mentions (`DISEASE`, `DRUG/CHEMICAL`, `SYMPTOM`, `PROCEDURE`, `TEST`, `ANATOMY`,
+ `DOSAGE`/`FREQUENCY`/`ROUTE`/`STRENGTH`/`DURATION`/`FORM`). (2) Assertion/negation status per entity. (3) Optional
normalization to CUI/SNOMED/RxNorm/ICD-10. Success = strong **entity-level F1 (seqeval)** per corpus, correct
negation handling, and a working linking demo.

## NHS Relevance

England's clinical record standard is **SNOMED CT**, with **ICD-10** and **OPCS-4** for secondary-care coding and
**dm+d** for medicines. Surfacing candidate SNOMED/ICD-10 concepts from a discharge summary supports coder
productivity, cohort/phenotype discovery, and structured secondary use. **Caveat:** almost all *public* annotated
clinical corpora are US English (MIMIC/i2b2/n2c2), so a real UK deployment faces genuine domain shift (spelling,
dm+d, SNOMED, note structure) — flagged as a first-class limitation.

## Business Value

NLP-assisted coding (pre-populate ICD-10/SNOMED/OPCS candidates → less manual abstraction, less under-coding) ·
cohort building/phenotyping from free text · ADE / pharmacovigilance signal detection · automated comorbidity/
procedure extraction for audit dashboards · reduced documentation burden.

## Expected Users

Clinical coders / CDI teams (assisted coding), researchers/epidemiologists (cohort discovery), medicines-safety
teams (ADE detection), and audit/operations (structured extraction) — all human-in-the-loop.

## Stakeholders

| Stakeholder | Value |
|---|---|
| Clinical coders / CDI | Candidate ICD-10/SNOMED/OPCS from narrative; faster, more complete coding |
| Researchers / epidemiologists | Find patients by free-text condition/medication/symptom patterns |
| Pharmacovigilance / medicines safety | Drug↔reaction linking; Yellow Card enrichment |
| Quality & audit | Automated comorbidity/procedure/outcome extraction |
| IG / governance | PHI kept inside boundary; structured, de-identifiable outputs |

## Dataset Options

**Openly downloadable (public build):**
1. **BC5CDR** ⭐ — 1,500 PubMed articles; **Chemical + Disease** spans with MeSH normalization + relations; open (HF/BigBIO).
2. **NCBI Disease Corpus** ⭐ — 793 abstracts; Disease spans + normalization; public domain.
3. **MACCROBAT** ⭐ — 200 clinical case reports; ~24 clinical entity/event types (symptom, diagnosis, medication,
   procedure, lab, anatomy, dosage, frequency, negation); brat standoff; open (figshare/CC). **Closest open proxy to clinical narrative.**
4. JNLPBA / BioNLP13CG / CRAFT (via scispaCy/BigBIO) — gene/protein/anatomy coverage; molecular-biology register.

**Credentialed / DUA-restricted (production path only, never public):**
5. **n2c2 2018 Track 2** (ADE + medication + relations, ~505 MIMIC-derived discharge summaries) — DBMI DUA.
6. **n2c2 / i2b2 2010** (Problem/Test/Treatment + assertions; 170 train / 256 test public subset) — DBMI DUA.
7. **i2b2 2009** (medication signature) — DBMI DUA.
8. **MIMIC-IV-Note v2.2** (331,794 de-identified discharge summaries — a *text source*, no labels) — **PhysioNet
   credentialed** (CITI + DUA). Basis of Bio_ClinicalBERT (MIMIC-III).

## Dataset Comparison

| Dataset | Size | Entity types | Register | Access |
|---|---|---|---|---|
| **BC5CDR** | 1,500 articles | Chemical, Disease (+norm, +relations) | Literature | **Open** |
| **NCBI-Disease** | 793 abstracts | Disease (+norm) | Literature | **Open** |
| **MACCROBAT** | 200 case reports | ~24 clinical types | **Clinical-ish** | **Open** |
| n2c2 2018 T2 | ~505 summaries | Drug + 7 attrs + ADE | Clinical EHR | DUA |
| i2b2 2010 | 170+256 | Problem/Test/Treatment | Clinical EHR | DUA |
| MIMIC-IV-Note | 331,794 notes | (unlabelled) | Clinical EHR | Credentialed |

**Hard separation:** MIMIC/i2b2/n2c2 (and any BigBIO wrapper of them) are credentialed/DUA — never commit their
text, derived annotations, or PHI-bearing model outputs to a public repo. BC5CDR/NCBI/MACCROBAT are safe for the public build.

## Recommended Dataset

**Public build:** **BC5CDR + NCBI-Disease + MACCROBAT.** BC5CDR gives high-quality Disease + Chemical/Drug spans
*with MeSH normalization* and relations; NCBI reinforces Disease NER against a well-known benchmark; MACCROBAT injects
genuine clinical register and a broad clinical schema. **Credentialed extension:** apply for **n2c2 2018 Track 2**
and **i2b2 2010**, backed by **MIMIC-IV-Note** for domain-adaptive pretraining — documented, kept private.

## Why This Dataset Was Selected

It is the strongest fully-**open** combination that covers the target entity families with gold spans and (for
BC5CDR/NCBI) concept normalization, while MACCROBAT supplies clinical-narrative style the literature corpora lack —
so the public repo is reproducible and shareable, with a credible credentialed path to real EHR text.

## Data Dictionary

Unified label schema mapping each corpus onto shared types: `DISEASE`, `DRUG/CHEMICAL`, `SYMPTOM`, `PROCEDURE`,
`TEST`, `ANATOMY`, `DOSAGE`, `FREQUENCY` (+ `STRENGTH`, `ROUTE`, `DURATION`, `FORM` for medication signature). Tags in
**IOB2** (`B-DISEASE`, `I-DISEASE`, `O`). Each entity carries: surface text, char offsets, type, confidence,
assertion/negation status, and optional normalized codes (`umls_cui`, `snomed_code`, `rxnorm_code`, `icd10_code`).
Report per-corpus results separately to stay comparable to published numbers.

## Data Challenges

Annotation-schema divergence across corpora (boundary/typing rules differ) · register mismatch (literature prose vs
telegraphic clinical notes) · abbreviations/acronyms ("MI", "PT", "cold" — context-dependent) · **negation &
assertion** ("no evidence of PE", "denies chest pain", "family history of", "rule out") · de-identification artifacts
(`[**Date**]` corrupt tokenization) · **US↔UK domain shift** (spelling, dm+d, SNOMED, structure) · long-tail/nested
entities · inter-annotator variation.

## Missing Values Strategy

Text analogue: enforce a canonical **IOB round-trip** (spans → tags → spans must be lossless); documents failing to
convert cleanly (misaligned brat offsets, overlapping spans) are quarantined, not silently dropped; missing
normalization codes stored as explicit `null` (unlinked), not fabricated.

## Feature Engineering Ideas

Contextual subword embeddings (not hand features). Design decisions instead: sentence splitting & tokenization
(scispaCy/medspaCy — clinical-aware), **wordpiece↔label alignment** via `word_ids()` (+`-100`), long-note
**sliding-window with overlap**, section detection, and a regex/gazetteer layer for dosage/frequency
(`\d+\s?mg`, `q6h`, `bid`) feeding the baseline.

## Exploratory Data Analysis Plan

Entity-type frequency per corpus · span-length distributions · abbreviation/acronym inventory · negation-cue
frequency · note/sentence length vs 512-token limit (windowing need) · label-alignment error audit ·
train/dev/test overlap & near-duplicate check · per-corpus schema mapping review.

## Machine Learning Strategy

Baseline (scispaCy + dictionary/regex) → fine-tuned transformer token-classifier (`AutoModelForTokenClassification`
via HF `Trainer`, `seqeval` in `compute_metrics`, early-stop on dev entity-F1) → medspaCy negation/assertion →
scispaCy/SapBERT entity linking → optional relation extraction. Train per-corpus and/or a unified multi-corpus model.

## Deep Learning Strategy (if applicable)

This is a transformer NLP project. Primary base **BiomedBERT (PubMedBERT, MIT)** for open literature corpora
(strongest reported on BC5CDR/NCBI); **Bio_ClinicalBERT** for clinical-register text (MACCROBAT + credentialed EHR).
Long-context **Clinical-Longformer** for whole-note reasoning (future). Correct wordpiece-label alignment is essential.

## Baseline Models

**scispaCy** `en_ner_bc5cdr_md` + curated gazetteer/regex — explainable floor, reuses scispaCy tokenizer and provides the UMLS linker.

## Advanced Models

Fine-tuned **BiomedBERT / Bio_ClinicalBERT** token-classifiers; **medspaCy ConText / negspacy** negation-assertion
pipe; **scispaCy `EntityLinker` (UMLS 2020AA)** or **SapBERT** dense linker → CUI → SNOMED/RxNorm/ICD-10 (dm+d for UK);
optional entity-marker BERT for medication-signature/ADE relations.

## Model Comparison Plan

Held-out test split per corpus; compare baseline vs BiomedBERT vs Bio_ClinicalBERT on **entity-level P/R/F1
(seqeval)** with per-type breakdown, **strict and relaxed** matching reported side by side; sanity-check against
published numbers (BiomedBERT ~0.87 F1 BC5CDR, ~0.88 NCBI); log to MLflow.

## Evaluation Metrics

**Entity-level P/R/F1 via `seqeval`** (primary — token-level accuracy is misleadingly inflated and wrong for NER) ·
per-entity-type F1 (`classification_report`) · **strict vs relaxed** matching (state which; use `nervaluate` for the
full matrix) · normalization accuracy@1 + candidate recall@k (given a correct span) · assertion/negation F1.

## Explainability Strategy

Structured **error taxonomy** first (type confusions, boundary/off-by-one, abbreviation false-negatives) · per-entity
confidence (softmax/CRF marginals; calibrate if used as a gate) · token attribution (Integrated Gradients / SHAP via
`transformers-interpret`/`captum`) with the "attention ≠ explanation" caveat · **human-in-the-loop** review of
low-confidence/high-impact entities (drug/allergy) with corrections captured as new training data (active learning).

## Risk Assessment

**PHI is the dominant risk.** Missing negation → reporting a ruled-out condition as present is a clinical-safety
failure. US-trained models degrade on NHS text — no real deployment without UK adaptation and subgroup evaluation.
The system **assists**; coders/clinicians decide.

## Ethical Considerations

Models trained on single-centre US data (MIMIC/BIDMC) encode demographic and care-pattern bias that does not
transfer cleanly to NHS populations; evaluate subgroup performance and document limitations. Never present
extraction as clinically authoritative without human oversight.

## Data Privacy Notes

**No redistribution** of MIMIC/n2c2/i2b2 text, derived annotations, or PHI-bearing outputs — not in the repo, model
cards, demos, artifacts, or third-party API calls (DUA/CITI conditions). Public build uses open corpora only; keep
credentialed work in a private access-controlled environment. Real NHS text → UK GDPR + DPA 2018, **Caldicott**,
**DSPT**, DPIA, legal basis, and **on-prem / TRE serving with no external PHI egress**; de-identification as a
pipeline stage with residual-PHI validation.

## Deployment Strategy

FastAPI (`/extract`) · Streamlit demo with **spaCy displaCy** entity rendering · PostgreSQL (extracted entities) ·
MLflow (tracking/registry) · Docker Compose · GitHub Actions. Serving: fine-tuned HF model (transformers pipeline or
ONNX for latency); batch + sliding-window for long notes. **Never load credentialed notes into a public demo.**

## API Design

`POST /extract`
```json
{ "text": "Pt started on metformin 500 mg BID for T2DM. No evidence of nephropathy." }
```
```json
{ "entities": [
    {"text":"metformin","start":14,"end":23,"label":"DRUG","score":0.99,"negated":false,
     "norm":{"rxnorm":"6809","umls_cui":"C0025598"}},
    {"text":"500 mg","start":24,"end":30,"label":"STRENGTH","score":0.97},
    {"text":"BID","start":31,"end":34,"label":"FREQUENCY","score":0.96},
    {"text":"T2DM","start":39,"end":43,"label":"DISEASE","score":0.95,"negated":false,
     "norm":{"snomed":"44054006","icd10":"E11","umls_cui":"C0011860"}},
    {"text":"nephropathy","start":59,"end":70,"label":"DISEASE","score":0.94,"negated":true,
     "norm":{"snomed":"90708001"}} ],
  "model_version":"biomedbert-ner-v1.3" }
```
Also `GET /health`, `POST /extract/batch`. Character offsets preserved end-to-end.

## Dashboard Ideas

Paste a note → **displaCy** highlighted entities colour-coded by type, negated entities visually distinguished;
per-entity confidence; normalized-code tooltips; export JSON. Disclaimer + no-credentialed-data banner.

## Database Design

```sql
CREATE TABLE documents (
  doc_id BIGSERIAL PRIMARY KEY, source TEXT, ingested_at TIMESTAMPTZ DEFAULT now());
CREATE TABLE entities (
  entity_id BIGSERIAL PRIMARY KEY, doc_id BIGINT REFERENCES documents,
  text TEXT NOT NULL, label TEXT NOT NULL, char_start INT NOT NULL, char_end INT NOT NULL,
  score REAL, negated BOOLEAN DEFAULT false, assertion TEXT,
  umls_cui TEXT, snomed_code TEXT, rxnorm_code TEXT, icd10_code TEXT,
  model_version TEXT, created_at TIMESTAMPTZ DEFAULT now());
CREATE INDEX idx_entities_label ON entities(label);
CREATE INDEX idx_entities_snomed ON entities(snomed_code);
```

## Folder Structure

```
05-clinical-nlp/
├── src/clinical_nlp/{config.py,data/,preprocess/,ner/,assertion/,linking/,api/,dashboard/}
├── tests/            # IOB round-trip, wordpiece alignment, seqeval metric, PHI-guard
├── data/             # gitignored (open corpora only; NO credentialed data)
├── notebooks/        # EDA only
├── docker/{Dockerfile,docker-compose.yml}
├── .github/workflows/ci.yml
├── Makefile ├─ pyproject.toml └─ README.md
```

## CI/CD Plan

GitHub Actions: lint (Ruff/Black) → unit tests (**IOB round-trip**, wordpiece-label alignment, seqeval on a fixed
open-corpus slice with an **F1 gate**, a **PHI-guard test** asserting no credentialed data paths) → Docker build.

## Testing Strategy

Unit tests for span↔IOB conversion (lossless), wordpiece alignment (`word_ids()`/`-100`), sliding-window stitching
(no entity split at boundaries), negation tagging, and metric correctness; integration test for `/extract` contract;
golden-file test on a pinned open-corpus sample; a guard test that credentialed corpora never enter the public pipeline.

## MLOps Plan

MLflow logs params + per-entity F1 + artefacts; model registry versions NER models; Docker Compose
(model + FastAPI + Streamlit + Postgres + MLflow); reproducible configs + seeds; model/data cards documenting corpora, metrics, and limitations.

## Monitoring Strategy

Latency/throughput, input-length distribution, entity-count drift, confidence-score drift, and (for real deployment)
periodic human-review agreement rates as a data-drift/quality signal; alert on F1 regression against the fixed eval slice.

## Future Improvements

Relation extraction at scale (drug→dose→frequency→reason→ADE; n2c2-2018 end-to-end) · UMLS/SNOMED linking at scale
(SapBERT/KRISSBERT dense linkers, abbreviation disambiguation, dm+d for UK) · **LLM-assisted extraction with
guardrails** (on-prem, strict JSON/constrained decoding, for rare types & pre-annotation — never unaudited, never PHI
to external APIs) · Clinical-Longformer whole-note reasoning · UK domain adaptation within a TRE + subgroup fairness ·
active learning from reviewer corrections.

## References

1. BC5CDR paper (PMC) — https://pmc.ncbi.nlm.nih.gov/articles/PMC4860626/ · HF — https://huggingface.co/datasets/masaenger/bc5cdr
2. NCBI Disease Corpus — https://www.ncbi.nlm.nih.gov/CBBresearch/Dogan/DISEASE/ · HF — https://huggingface.co/datasets/ncbi/ncbi_disease
3. MACCROBAT — https://figshare.com/collections/MACCROBAT/4652765
4. n2c2 2018 Track 2 (ADE & Medication) — https://n2c2.dbmi.hms.harvard.edu/challenge/2018-track-2-ade-medication-extraction
5. n2c2 / i2b2 2010 — https://portal.dbmi.hms.harvard.edu/projects/n2c2-2010/
6. MIMIC-IV-Note v2.2 (PhysioNet, credentialed) — https://physionet.org/content/mimic-iv-note/2.2/

## Useful Research Links

7. Bio_ClinicalBERT — https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT
8. BioBERT v1.1 — https://huggingface.co/dmis-lab/biobert-v1.1
9. PubMedBERT / BiomedBERT (MIT) — https://huggingface.co/microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext
10. scispaCy — https://github.com/allenai/scispacy · models — https://allenai.github.io/scispacy/
11. seqeval — https://github.com/chakki-works/seqeval · nervaluate — https://github.com/MantisAI/nervaluate
12. medspaCy (ConText negation/assertion) — https://github.com/medspacy/medspacy
13. HF token-classification guide — https://huggingface.co/docs/transformers/tasks/token_classification
14. BigBIO (loaders; gated data still gated) — https://github.com/bigscience-workshop/biomedical
15. spaCy displaCy — https://spacy.io/usage/visualizers
16. Uzuner et al. 2010 i2b2/VA challenge, JAMIA — https://academic.oup.com/jamia/article/18/5/552/830538

---

*Public build on open corpora (BC5CDR + NCBI-Disease + MACCROBAT); BiomedBERT/Bio_ClinicalBERT token-classifiers with
correct wordpiece alignment; entity-level seqeval (strict + relaxed); medspaCy negation + scispaCy UMLS linking.
MIMIC/n2c2/i2b2 = credentialed, non-redistributable extension. PHI/DUA/UK-IG handling is the central constraint.*
