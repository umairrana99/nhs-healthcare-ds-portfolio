"""Engineered clinical features for the readmission model.

Each function takes and returns a DataFrame (adding columns, never mutating the
input) so steps can be composed and unit-tested in isolation. ``engineer_features``
runs them all. The choices here follow the design doc's "Feature Engineering"
section — the strongest signals on this dataset are prior utilisation, comorbidity
burden, and medication complexity.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from readmission.constants import MEDICATION_COLUMNS

# A1C and serum-glucose results are ordinal; "None" means the test was not run,
# which we capture separately as a "measured" flag (its absence is informative).
_A1C_LEVELS = {"Norm": 1, ">7": 2, ">8": 3}
_GLUCOSE_LEVELS = {"Norm": 1, ">200": 2, ">300": 3}
_MED_ACTIVE_VALUES = frozenset({"Steady", "Up", "Down"})


def add_prior_utilisation(df: pd.DataFrame) -> pd.DataFrame:
    """Total prior-year visits and a binary 'any prior inpatient' flag."""
    out = df.copy()
    cols = ["number_outpatient", "number_emergency", "number_inpatient"]
    out["total_prior_visits"] = out[cols].sum(axis=1)
    out["any_prior_inpatient"] = (out["number_inpatient"] > 0).astype("int8")
    return out


def add_medication_features(df: pd.DataFrame) -> pd.DataFrame:
    """Count of active diabetic medications and a polypharmacy flag."""
    out = df.copy()
    present = [c for c in MEDICATION_COLUMNS if c in out.columns]
    out["active_med_count"] = out[present].isin(_MED_ACTIVE_VALUES).sum(axis=1).astype("int16")
    if "num_medications" in out.columns:
        out["polypharmacy"] = (out["num_medications"] > 15).astype("int8")
    return out


def _ordinal(series: pd.Series, mapping: dict[str, int]) -> pd.Series:
    """Map values via ``mapping``; anything else (incl. 'None'/NA) becomes 0."""
    return series.map(mapping).fillna(0).astype("int8")


def add_glycemic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ordinal A1C / glucose levels plus 'was it measured' flags."""
    out = df.copy()
    if "A1Cresult" in out.columns:
        out["a1c_measured"] = out["A1Cresult"].notna() & out["A1Cresult"].ne("None")
        out["a1c_measured"] = out["a1c_measured"].astype("int8")
        out["a1c_level"] = _ordinal(out["A1Cresult"], _A1C_LEVELS)
    if "max_glu_serum" in out.columns:
        out["glucose_measured"] = out["max_glu_serum"].notna() & out["max_glu_serum"].ne("None")
        out["glucose_measured"] = out["glucose_measured"].astype("int8")
        out["glucose_level"] = _ordinal(out["max_glu_serum"], _GLUCOSE_LEVELS)
    return out


def _age_band_midpoint(band: object) -> float:
    """Convert an age band like ``"[70-80)"`` to its midpoint (75.0)."""
    if pd.isna(band):
        return float("nan")
    digits = str(band).strip("[]()").split("-")
    try:
        low, high = int(digits[0]), int(digits[1])
    except (ValueError, IndexError):
        return float("nan")
    return (low + high) / 2.0


def add_age_midpoint(df: pd.DataFrame) -> pd.DataFrame:
    """Numeric age from the 10-year band (e.g. ``"[70-80)"`` -> 75)."""
    out = df.copy()
    if "age" in out.columns:
        out["age_midpoint"] = out["age"].map(_age_band_midpoint).astype(np.float64)
    return out


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all engineered-feature steps in order."""
    out = add_prior_utilisation(df)
    out = add_medication_features(out)
    out = add_glycemic_features(out)
    out = add_age_midpoint(out)
    return out
