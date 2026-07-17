"""Entry point: train, evaluate, and save the readmission model.

Run with the dataset present (see ``readmission.data.download``):

    python -m readmission.train

Compares the logistic-regression baseline against XGBoost with grouped
cross-validation, then fits XGBoost on all data and saves it for the API to serve.
"""

from __future__ import annotations

from readmission.config import get_settings
from readmission.constants import PATIENT_ID, TARGET_COLUMN
from readmission.data.ingest import build_dataset, load_raw
from readmission.evaluate import compare_models
from readmission.logging_utils import configure_logging, get_logger
from readmission.models.baseline import build_baseline_pipeline
from readmission.models.boosted import build_xgboost_pipeline
from readmission.persist import save_model
from readmission.preprocess import select_model_frame

logger = get_logger(__name__)


def main() -> dict[str, dict[str, float]]:
    """Compare models, save the fitted XGBoost model, and return the metrics."""
    settings = get_settings()
    configure_logging(settings.log_level)

    clean = build_dataset(
        load_raw(settings.raw_data_path),
        first_encounter_only=settings.first_encounter_only,
    )
    frame = select_model_frame(clean)
    y = clean[TARGET_COLUMN].to_numpy()
    groups = clean[PATIENT_ID].to_numpy()

    comparison = compare_models(
        frame,
        y,
        groups,
        {"logreg": build_baseline_pipeline, "xgboost": build_xgboost_pipeline},
        random_state=settings.random_seed,
    )
    for name, metrics in comparison.items():
        logger.info(
            "%s grouped-CV: AUROC %.3f | AUPRC %.3f | Brier %.3f | ECE %.3f",
            name,
            metrics["auroc_mean"],
            metrics["auprc_mean"],
            metrics["brier_mean"],
            metrics["ece_mean"],
        )

    model = build_xgboost_pipeline(random_state=settings.random_seed).fit(frame, y)
    save_model(model, settings.model_path)
    logger.info("Saved model to %s", settings.model_path)
    return comparison


if __name__ == "__main__":
    main()
