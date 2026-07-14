"""Load and clean the UCI Diabetes 130-US Hospitals dataset.

The public entry point is :func:`build_dataset`, which composes the individual
cleaning steps. Each step is a small, pure function so it can be unit-tested in
isolation — the leakage-prevention rules in particular have dedicated tests.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from readmission.constants import (
    DISCHARGE_DISPOSITION_COLUMN,
    ENCOUNTER_ID,
    EXPIRED_HOSPICE_DISPOSITION_IDS,
    MISSING_TOKEN,
    PATIENT_ID,
    POSITIVE_RAW_VALUE,
    RAW_TARGET_COLUMN,
    TARGET_COLUMN,
)
from readmission.logging_utils import get_logger

logger = get_logger(__name__)


def load_raw(path: Path) -> pd.DataFrame:
    """Read the raw CSV, converting the literal ``"?"`` token to missing values."""
    if not path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {path}. "
            "Download it first (see readmission.data.download.get_download_instructions)."
        )
    df = pd.read_csv(path, na_values=[MISSING_TOKEN], low_memory=False)
    logger.info("Loaded raw dataset: %d rows x %d columns", len(df), df.shape[1])
    return df


def replace_missing_tokens(df: pd.DataFrame) -> pd.DataFrame:
    """Replace the literal ``"?"`` token with proper missing values.

    Redundant after :func:`load_raw` (which handles it at read time) but kept as
    a standalone step so in-memory frames and tests are cleaned consistently.
    """
    return df.replace(MISSING_TOKEN, pd.NA)


def drop_expired_hospice(df: pd.DataFrame) -> pd.DataFrame:
    """Remove encounters ending in death or hospice transfer.

    These patients cannot be readmitted; keeping them leaks the outcome and
    biases the negative class. This is the single most common mistake on this
    dataset (see design doc, "Data Challenges").
    """
    if DISCHARGE_DISPOSITION_COLUMN not in df.columns:
        raise KeyError(f"Expected column '{DISCHARGE_DISPOSITION_COLUMN}' not found.")
    mask = df[DISCHARGE_DISPOSITION_COLUMN].isin(EXPIRED_HOSPICE_DISPOSITION_IDS)
    removed = int(mask.sum())
    logger.info("Dropping %d expired/hospice encounters (cannot be readmitted)", removed)
    return df.loc[~mask].copy()


def binarize_target(df: pd.DataFrame, *, drop_raw: bool = True) -> pd.DataFrame:
    """Add the binary 30-day target: 1 if readmitted within 30 days, else 0.

    Only the raw value ``"<30"`` is positive; ``">30"`` and ``"NO"`` are 0.
    """
    if RAW_TARGET_COLUMN not in df.columns:
        raise KeyError(f"Expected target column '{RAW_TARGET_COLUMN}' not found.")
    out = df.copy()
    out[TARGET_COLUMN] = (out[RAW_TARGET_COLUMN] == POSITIVE_RAW_VALUE).astype("int8")
    if drop_raw:
        out = out.drop(columns=[RAW_TARGET_COLUMN])
    return out


def keep_first_encounter_per_patient(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only each patient's earliest encounter.

    Makes rows statistically independent (Strack et al.). ``encounter_id`` is
    monotonic in time, so the smallest id per patient is the first encounter.
    """
    if PATIENT_ID not in df.columns or ENCOUNTER_ID not in df.columns:
        raise KeyError(f"Expected id columns '{PATIENT_ID}' and '{ENCOUNTER_ID}'.")
    before = len(df)
    out = df.sort_values(ENCOUNTER_ID).drop_duplicates(subset=PATIENT_ID, keep="first")
    logger.info("First-encounter filter: %d -> %d rows", before, len(out))
    return out.reset_index(drop=True)


def build_dataset(df: pd.DataFrame, *, first_encounter_only: bool = True) -> pd.DataFrame:
    """Run the full cleaning pipeline on a raw dataframe.

    Steps: normalise missing tokens -> drop death/hospice -> (optional) keep
    first encounter per patient -> binarise the 30-day target.
    """
    out = replace_missing_tokens(df)
    out = drop_expired_hospice(out)
    if first_encounter_only:
        out = keep_first_encounter_per_patient(out)
    out = binarize_target(out)
    logger.info(
        "Built dataset: %d rows, positive rate %.3f",
        len(out),
        float(out[TARGET_COLUMN].mean()) if len(out) else 0.0,
    )
    return out
