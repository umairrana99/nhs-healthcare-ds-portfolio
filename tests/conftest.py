"""Shared test fixtures.

The fixture mirrors the real schema on a handful of rows so the cleaning logic
can be tested without downloading the licensed dataset.
"""

from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture
def raw_df() -> pd.DataFrame:
    """A tiny raw-shaped frame covering the cases the cleaning logic must handle.

    Rows:
      1 (patient 100) survives, readmitted "<30"  -> positive, kept
      2 (patient 100) later encounter, "NO"       -> dropped by first-encounter filter
      3 (patient 200) expired (disposition 11)     -> dropped (death)
      4 (patient 300) hospice (disposition 13)     -> dropped (hospice)
      5 (patient 400) readmitted ">30"             -> negative, kept
      6 (patient 500) has "?" race + ">30"         -> "?" -> NA, negative, kept
    """
    return pd.DataFrame(
        {
            "encounter_id": [1, 2, 3, 4, 5, 6],
            "patient_nbr": [100, 100, 200, 300, 400, 500],
            "race": ["Caucasian", "Caucasian", "AfricanAmerican", "Caucasian", "Asian", "?"],
            "discharge_disposition_id": [1, 1, 11, 13, 1, 1],
            "readmitted": ["<30", "NO", "<30", "NO", ">30", ">30"],
        }
    )
