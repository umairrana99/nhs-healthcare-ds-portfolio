"""Group ICD-9 diagnosis codes into clinical categories.

The ``diag_1/2/3`` columns hold ~700 distinct ICD-9 codes. One-hot encoding them
raw explodes the feature space and dilutes signal, so we collapse them into nine
clinically meaningful buckets following Strack et al. (2014) — the grouping used
in the paper that published this dataset.
"""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

# Ordered set of possible outputs (useful for consistent categorical encoding).
DIAGNOSIS_GROUPS: tuple[str, ...] = (
    "Circulatory",
    "Respiratory",
    "Digestive",
    "Diabetes",
    "Injury",
    "Musculoskeletal",
    "Genitourinary",
    "Neoplasms",
    "Other",
    "Missing",
)


def group_icd9(code: object) -> str:
    """Map a single ICD-9 code to its clinical group.

    Diabetes (``250.xx``) is checked first. ``V``/``E`` supplementary codes and
    anything outside the mapped ranges fall through to ``"Other"``; missing
    values become ``"Missing"``.
    """
    if pd.isna(code):
        return "Missing"

    text = str(code).strip()
    if not text:
        return "Missing"
    if text[0] in {"V", "v", "E", "e"}:
        return "Other"
    if text.startswith("250"):
        return "Diabetes"

    try:
        value = int(float(text))
    except ValueError:
        return "Other"

    if 390 <= value <= 459 or value == 785:
        return "Circulatory"
    if 460 <= value <= 519 or value == 786:
        return "Respiratory"
    if 520 <= value <= 579 or value == 787:
        return "Digestive"
    if 800 <= value <= 999:
        return "Injury"
    if 710 <= value <= 739:
        return "Musculoskeletal"
    if 580 <= value <= 629 or value == 788:
        return "Genitourinary"
    if 140 <= value <= 239:
        return "Neoplasms"
    return "Other"


def add_diagnosis_groups(
    df: pd.DataFrame,
    columns: Iterable[str] = ("diag_1", "diag_2", "diag_3"),
) -> pd.DataFrame:
    """Return a copy of ``df`` with a ``<col>_group`` column for each diag column."""
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[f"{col}_group"] = out[col].map(group_icd9)
    return out
