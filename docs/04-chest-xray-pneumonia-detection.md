# 04 — Chest X-ray Pneumonia Detection System (with Grad-CAM)

> **Design document.** Detect radiographic pneumonia on frontal chest X-rays using deep learning, with
> **Grad-CAM explainability that is objectively validated** against annotated opacity regions. A research/
> portfolio decision-support tool — **not a certified medical device, not autonomous diagnosis.**

---

## Executive Summary

The chest radiograph is the first-line imaging test for pneumonia, and NHS radiology reporting is under sustained
workforce pressure. This project builds a CXR pneumonia classifier framed as **triage/second-read decision
support** with a saliency overlay. The distinguishing quality move: rather than showing a pretty heatmap, it
**measures** whether Grad-CAM lands inside the lungs on the annotated opacity (IoU / hit-rate against RSNA
bounding boxes) and runs saliency **sanity checks** — turning "explainability" from a demo into a verified claim.
Build order: a fast, clean **Kermany** binary baseline (re-split to fix the leaky 16-image val set), then an
adult, bounding-box-validated **RSNA** upgrade. Safety framing (high sensitivity, human-in-the-loop, regulatory
disclaimer) is explicit throughout.

## Problem Statement

Binary detection of radiographic pneumonia (Normal vs Pneumonia) on frontal CXR, delivered as CAD triage. Success
= high **sensitivity** at a safety-driven operating point with strong AUROC/PR-AUC, **plus** Grad-CAM that
demonstrably attends to lung opacity (not tokens/corners). Beating a leaky-split "99% accuracy" notebook is *not* success.

## NHS Relevance

The Royal College of Radiologists reports a large, growing consultant-radiologist shortfall and reporting
backlogs, with plain radiographs (incl. CXR) the highest-volume workload. A tool that (a) prioritises likely-abnormal
films in the queue and (b) speeds second-read via a saliency overlay is a realistic assistive use case — analogous
to deployed CE/UKCA-marked CXR triage products. **Regulatory reality:** software informing a clinical decision is
a medical device (UKCA under UK MDR 2002 / MHRA; CE under EU MDR 2017/745), typically Class IIa/IIb, requiring
ISO 13485, clinical evaluation and post-market surveillance. **This project does none of that and carries a
prominent "Research/education only — not a medical device" disclaimer everywhere.**

## Business Value

Worklist prioritisation (abnormal-first), a fast visual second-read reducing perceptual misses, faster turnaround
and lower backlog, and out-of-hours provisional reads in ED chest-infection/sepsis pathways. Value is realised
**only as assistance** — the safety case rests on high sensitivity plus radiologist oversight.

## Expected Users

Reporting radiologists (worklist + second-read), ED clinicians (provisional out-of-hours read), and triage/operations
(turnaround KPIs) — always with a human decision-maker.

## Stakeholders

| Stakeholder | Value |
|---|---|
| Reporting radiologists | Prioritise likely-abnormal; focus attention via Grad-CAM |
| ED clinicians | Faster provisional read when report delayed; triage support |
| Triage / operations | Reduced turnaround time and backlog |
| Trust / commissioners | Workforce multiplier; auditable, explainable outputs |
| Patients | Faster time-to-treatment for true pneumonia |

## Dataset Options

1. **Kermany "Chest X-Ray Images (Pneumonia)"** ⭐ baseline — 5,863 JPEGs (train 5,216 / val 16 / test 624),
   `NORMAL`/`PNEUMONIA`, **CC BY 4.0**; **paediatric (1–5y), single centre**; known tiny/leaky val set & near-duplicates.
2. **RSNA Pneumonia Detection Challenge 2018** ⭐ upgrade — ~26,684 adult frontal CXRs (DICOM) with **bounding
   boxes**; curated from NIH; competition/academic terms.
3. **NIH ChestX-ray14** — 112,120 images / 30,805 patients; 14 NLP-mined (noisy) labels; the classic shortcut-learning cautionary dataset.
4. **CheXpert (Stanford)** — 224,316 CXRs; uncertainty labels; registration + research agreement.
5. **MIMIC-CXR / -JPG** — 377,110 images + reports; **PhysioNet credentialed**; overkill for a baseline, great for multimodal future work.

## Dataset Comparison

| Dataset | N | Population | Format | Localization labels | Access | Best for |
|---|---:|---|---|---|---|---|
| **Kermany** | 5,863 | Paediatric | JPEG | No | Open (CC BY 4.0) | Clean binary baseline |
| **RSNA 2018** | ~26,684 | Adult | DICOM | **Yes (bboxes)** | Comp. rules | Model + Grad-CAM validation |
| NIH CXR14 | 112,120 | Adult | PNG | Subset | Low | Multi-label, shortcut study |
| CheXpert | 224,316 | Adult | JPG | No | Registration | Large benchmark |
| MIMIC-CXR | 377,110 | Adult | JPG/DICOM | No | Credentialed | Multimodal/reports |

## Recommended Dataset

**Two-tier:** (1) **Kermany** first — permissive licence, folder-structured, small → a fully reproducible
end-to-end pipeline in days (re-split it, report the paediatric limitation honestly). (2) **RSNA 2018** as the
credible upgrade — adult population (NHS-relevant) and, critically, **bounding boxes let you objectively validate
Grad-CAM** (IoU/hit-rate vs annotated opacity).

## Why This Dataset Was Selected

Kermany buys speed and a clean binary signal to prove the pipeline; RSNA buys adult relevance and **measured**
explainability — the single biggest differentiator versus the ubiquitous "99% on the leaky Kermany split"
notebook. The trade-off (DICOM engineering + label ambiguity) is worth it for a spatially-verifiable, NHS-credible result.

## Data Dictionary

Kermany: JPEG, variable resolution, grayscale; train NORMAL 1,341 / PNEUMONIA 3,875 (~3:1), test 234/390, val 8/8.
RSNA: 1024×1024 DICOM, one frontal view/patient, ~77% negative / 23% opacity-positive; a 3-class metadata field
(Normal / No Lung Opacity–Not Normal / Lung Opacity) + bounding boxes for positives. X-rays are single-channel
(replicate to 3 for ImageNet backbones); honour DICOM windowing & MONOCHROME1 inversion.

## Data Challenges

Class imbalance (~3:1 Kermany, ~1:3 RSNA) · **shortcut/confounder learning** (Zech 2018: CNNs learn hospital
tokens/corner artefacts, predicting provenance not pathology) · **tiny/leaky Kermany val set (16 images)** —
discard the canonical split · **paediatric→adult domain shift** · label noise (NLP-mined; "opacity" ≠ confirmed
pneumonia) · **patient-level leakage** (split by patient, never image; de-dup).

## Missing Values Strategy

Not a tabular-missingness problem. Analogue: reject corrupt/non-frontal/lateral images at ingest; validate each
DICOM has readable pixel data + required tags; drop duplicates (perceptual hash) so they can't cross splits.

## Feature Engineering Ideas

Learned representations, not hand features. The equivalent decisions are **preprocessing & augmentation**:
resize to 224×224 (or 224–300 for EfficientNet), ImageNet normalisation, grayscale→3-channel; anatomy-preserving
augmentation only (small rotations ±5–10°, mild translate/scale/shear, light brightness/contrast/noise);
**no vertical flips**; **horizontal flip off by default** (laterality matters); optional CLAHE (evaluate); light TTA.

## Exploratory Data Analysis Plan

Class balance per split (confirm no patient leakage) · resolution/aspect/intensity histograms per class (intensity
shortcuts) · sample grids inspecting for markers/tokens/borders · **mean-image-per-class + difference map** (reveals
token/label shortcuts before training) · RSNA bbox count/size distribution + overlays · perceptual-hash duplicate check.

## Machine Learning Strategy

Framework **PyTorch** (torchvision backbones, first-class `pytorch-grad-cam`/`torchcam`, pydicom/MONAI ecosystem).
Baseline simple CNN → transfer learning; class weighting / weighted sampler / focal loss; two-stage fine-tuning
(freeze→unfreeze low-LR, discriminative LRs); early-stop on val AUROC/recall; mixed precision; fixed seeds + logged configs.

## Deep Learning Strategy (if applicable)

This is a deep-learning project. Compare ImageNet-pretrained **DenseNet121** (CheXNet backbone), **ResNet50**
(well-understood Grad-CAM), **EfficientNet-B0/B3** (accuracy/compute). Single-logit binary head; BCE-with-logits +
class weighting; cosine/ReduceLROnPlateau; light test-time augmentation.

## Baseline Models

A small 3–4-block CNN trained from scratch (establishes a floor and validates the pipeline) — before any pretrained backbone.

## Advanced Models

Transfer-learned **DenseNet121 / ResNet50 / EfficientNet** with fine-tuning; the chosen model calibrated
(temperature scaling) and thresholded to a target recall.

## Model Comparison Plan

Fixed patient-wise train/val/test split (seeded); compare backbones on AUROC (bootstrap CI), PR-AUC, and
sensitivity at the safety operating point; on RSNA additionally compare **Grad-CAM localization hit-rate/IoU**;
where feasible, cross-dataset test (paediatric→adult) to expose domain shift; log to MLflow.

## Evaluation Metrics

**AUROC** (primary, 95% CI) · **sensitivity/recall PRIORITISED** — choose the threshold to hit a target recall
(e.g. ≥0.95) and report resulting specificity (explicit safety operating point, not 0.5) · specificity, PPV/NPV
(NPV matters for rule-out) · **PR-AUC** · F1 + confusion matrix at the operating point · calibration (reliability,
ECE) · **Grad-CAM localization IoU/hit-rate vs bounding boxes** as an explainability metric. **Accuracy alone is unacceptable.**

## Explainability Strategy

**Grad-CAM / Grad-CAM++** overlays on the last conv block (DenseNet `features.norm5` / ResNet `layer4`) via
`pytorch-grad-cam` or `torchcam` (also Score-CAM/Eigen-CAM for cross-checking). **Mandatory sanity checks:**
(1) **localization** — quantify Grad-CAM ∩ RSNA bounding box (IoU/hit-rate); (2) **confounder** — verify the model
is *not* attending to corners/markers/text (Zech failure mode); (3) **cross-method agreement**; (4) **saliency
sanity checks** (Adebayo 2018 — randomising weights should destroy the map); (5) a curated **failure gallery**.
Deliverable: every prediction returns an original|overlay pair.

## Risk Assessment

No autonomous diagnosis; radiologist-in-the-loop; disclaimer everywhere. Failure modes: **missed pneumonia**
(highest harm), over-flagging (alert fatigue), OOD inputs (laterals, paediatric vs adult, different scanner),
shortcut learning, subgroup degradation. Automation bias — present as a *suggestion with evidence*, show confidence,
consider abstention/OOD detection ("low confidence — refer to radiologist").

## Ethical Considerations

Dataset bias (paediatric-only Kermany; single-centre) → never claim general validity; state population/provenance.
Report performance by age/sex if metadata available. Regulatory framing repeated (research-only). Over-reliance and deskilling risks.

## Data Privacy Notes

Public datasets are HIPAA-deidentified. In the app: **strip DICOM PHI headers on ingest**, never log raw images
with identifiers, don't persist uploads beyond the request unless consented; state UK-GDPR relevance for any real data.

## Deployment Strategy

FastAPI (multipart image upload) · Streamlit demo · MLflow (registry) · Docker (CPU default, GPU optional) ·
GitHub Actions · drift monitoring. Load model once at startup (lifespan); CPU inference is fast for a single CXR; batch endpoint for worklist mode.

## API Design

`POST /predict` (multipart `file=<xray.jpg|.png|.dcm>`)
```json
{ "prediction":"PNEUMONIA","probability":0.9124,"threshold":0.35,"operating_point":"recall>=0.95",
  "gradcam_png_base64":"iVBORw0KGgo...","model_version":"densenet121-rsna-v1.2.0",
  "disclaimer":"Research use only. Not a medical device." }
```
Handler: load image (JPEG/PNG/DICOM + PHI strip) → preprocess (resize, ImageNet-normalise, 3-ch) → sigmoid logit →
label at safety threshold → Grad-CAM overlay (base64 PNG). Validate file type/size/single-frontal-view; request size limits. Also `GET /health`, `POST /predict/batch`.

## Dashboard Ideas

Upload an X-ray → show original, predicted class + probability + operating point, and the **Grad-CAM heatmap
side-by-side**; disclaimer banner; confidence/OOD warning. (`st.file_uploader` → FastAPI or in-process model.)

## Database Design

```sql
CREATE TABLE studies (
  study_id BIGSERIAL PRIMARY KEY, source TEXT, view TEXT, ingested_at TIMESTAMPTZ DEFAULT now());
CREATE TABLE predictions (
  prediction_id BIGSERIAL PRIMARY KEY, study_id BIGINT REFERENCES studies,
  model_version TEXT, prediction TEXT, probability NUMERIC(5,4), threshold_used NUMERIC(5,4),
  operating_point TEXT, gradcam_uri TEXT, created_at TIMESTAMPTZ DEFAULT now());
CREATE TABLE radiologist_feedback (
  id BIGSERIAL PRIMARY KEY, prediction_id BIGINT REFERENCES predictions,
  confirmed_label TEXT, created_at TIMESTAMPTZ DEFAULT now());
```
Store Grad-CAM as a URI to object storage, never raw identifiable images in-DB.

## Folder Structure

```
04-pneumonia-detection/
├── src/pneumonia/{config.py,data/,transforms/,models/,train/,explain/,api/,dashboard/}
├── tests/            # preprocessing, DICOM PHI strip, endpoint contract, "grad-cam produces a map"
├── data/             # gitignored
├── notebooks/        # EDA only
├── docker/{Dockerfile.cpu,Dockerfile.gpu,docker-compose.yml}
├── .github/workflows/ci.yml
├── Makefile ├─ pyproject.toml └─ README.md
```

## CI/CD Plan

GitHub Actions: lint (Ruff/Black) → unit tests (preprocessing, **DICOM PHI strip**, endpoint contract, Grad-CAM
smoke test) → build & smoke-test the **CPU Docker image** → optional push. Data/schema test on a tiny sample.

## Testing Strategy

Unit tests for image loading (JPEG/PNG/DICOM, MONOCHROME1 inversion, PHI strip), transforms (shape/normalisation),
threshold logic, and a Grad-CAM "returns a valid heatmap on the right layer" test; integration test for `/predict`
(multipart) with a fixture image; a small regression eval with an AUROC/recall floor.

## MLOps Plan

MLflow tracks params + AUROC/recall/PR-AUC + artefacts (confusion matrix, Grad-CAM samples, localization IoU);
Model Registry versions the chosen model with a stage; API serves the registry URI; Docker ships code+model, not data.

## Monitoring Strategy

Log prediction distributions, input-property drift (image size/intensity), score drift, flag rate, and OOD/low-confidence
rate; latency/throughput; radiologist-confirmation feedback loop → performance monitoring. Tools: Prometheus/Grafana or Evidently.

## Future Improvements

Multi-label (CheXpert/NIH 14 findings) · move from classification to **detection/segmentation** on RSNA (RetinaNet/
YOLO/U-Net, report mAP) · **external validation** across datasets to quantify Zech shortcut risk · calibration
(temperature/isotonic) + conformal thresholds · OOD/abstention module · lung-field segmentation preprocessing ·
fairness audit · self-supervised pretraining (MIMIC-CXR) · report-generation multimodal extension.

## References

1. Kermany CXR (Kaggle) — https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
2. Kermany primary data (Mendeley) — https://data.mendeley.com/datasets/rscbjbr9sj/3
3. RSNA Pneumonia Detection Challenge (Kaggle) — https://www.kaggle.com/c/rsna-pneumonia-detection-challenge
4. NIH ChestX-ray14 — https://nihcc.app.box.com/v/ChestXray-NIHCC
5. CheXpert (Stanford) — https://stanfordmlgroup.github.io/competitions/chexpert/
6. Kermany et al., Cell 2018 — https://doi.org/10.1016/j.cell.2018.02.010

## Useful Research Links

7. Wang et al., ChestX-ray8/14 (arXiv:1705.02315) — https://arxiv.org/abs/1705.02315
8. Rajpurkar et al., CheXNet (arXiv:1711.05225) — https://arxiv.org/abs/1711.05225
9. **Zech et al., confounding/generalization, PLOS Med** — https://journals.plos.org/plosmedicine/article?id=10.1371/journal.pmed.1002683
10. Selvaraju et al., Grad-CAM (arXiv:1610.02391) — https://arxiv.org/abs/1610.02391
11. Chattopadhyay et al., Grad-CAM++ (arXiv:1710.11063) — https://arxiv.org/abs/1710.11063
12. Adebayo et al., Sanity Checks for Saliency Maps (arXiv:1810.03292) — https://arxiv.org/abs/1810.03292
13. `pytorch-grad-cam` — https://github.com/jacobgil/pytorch-grad-cam · `torchcam` — https://github.com/frgfm/torch-cam
14. torchvision models — https://pytorch.org/vision/stable/models.html · MONAI — https://monai.io/ · pydicom — https://pydicom.github.io/
15. MHRA Software & AI as a Medical Device — https://www.gov.uk/government/publications/software-and-artificial-intelligence-ai-as-a-medical-device
16. EU MDR 2017/745 — https://eur-lex.europa.eu/eli/reg/2017/745/oj

---

*Ship Kermany (clean baseline, re-split, honest paediatric caveat) → upgrade to RSNA (adult + bounding-box-validated
Grad-CAM). The measured-localization + domain-shift + regulatory framing is what makes this read as rigorous, not a toy.*
