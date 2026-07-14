"""Entry point: train and cross-validate the baseline readmission model.

Run with the dataset present (see ``readmission.data.download``):

    python -m readmission.train
"""

from __future__ import annotations

from readmission.config import get_settings
from readmission.constants import PATIENT_ID, TARGET_COLUMN
from readmission.data.ingest import build_dataset, load_raw
from readmission.evaluate import cross_validate
from readmission.logging_utils import configure_logging, get_logger
from readmission.preprocess import select_model_frame

logger = get_logger(__name__)


def main() -> dict[str, float]:
    """Load data, build features, and report baseline grouped-CV metrics."""
    settings = get_settings()
    configure_logging(settings.log_level)

    clean = build_dataset(
        load_raw(settings.raw_data_path),
        first_encounter_only=settings.first_encounter_only,
    )
    frame = select_model_frame(clean)
    y = clean[TARGET_COLUMN].to_numpy()
    groups = clean[PATIENT_ID].to_numpy()

    metrics = cross_validate(frame, y, groups, random_state=settings.random_seed)
    logger.info(
        "Baseline (LogReg) grouped-CV: AUROC %.3f ± %.3f | AUPRC %.3f ± %.3f",
        metrics["auroc_mean"],
        metrics["auroc_std"],
        metrics["auprc_mean"],
        metrics["auprc_std"],
    )
    return metrics


if __name__ == "__main__":
    main()
