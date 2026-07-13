"""Domain constants for the UCI Diabetes 130-US Hospitals dataset.

Centralising these avoids magic values scattered across the pipeline and gives a
single place to document the leakage-prevention decisions from the design doc.
"""

from __future__ import annotations

# Raw CSV encodes missing values as a literal "?".
MISSING_TOKEN = "?"

# Identifier columns. ``patient_nbr`` is critical: a patient recurs across many
# encounters, so any train/test split MUST group on it to avoid leakage.
ENCOUNTER_ID = "encounter_id"
PATIENT_ID = "patient_nbr"
ID_COLUMNS: tuple[str, ...] = (ENCOUNTER_ID, PATIENT_ID)

# Target. Raw ``readmitted`` is one of {"<30", ">30", "NO"}. For a 30-day model
# only "<30" is positive; folding ">30" into the positive class answers a
# different question (see design doc, "Data Dictionary").
RAW_TARGET_COLUMN = "readmitted"
TARGET_COLUMN = "readmitted_30d"
POSITIVE_RAW_VALUE = "<30"

# Discharge dispositions that describe patients who CANNOT be readmitted:
# death and hospice transfers. Leaving these rows in leaks the outcome and
# biases the negative class — every serious analysis removes them.
#   11 Expired
#   13 Hospice / home
#   14 Hospice / medical facility
#   19 Expired at home (Medicaid hospice)
#   20 Expired in medical facility (Medicaid hospice)
#   21 Expired, place unknown (Medicaid hospice)
EXPIRED_HOSPICE_DISPOSITION_IDS: frozenset[int] = frozenset({11, 13, 14, 19, 20, 21})
DISCHARGE_DISPOSITION_COLUMN = "discharge_disposition_id"

# The 23 medication columns, each ∈ {"No", "Steady", "Up", "Down"}.
MEDICATION_COLUMNS: tuple[str, ...] = (
    "metformin",
    "repaglinide",
    "nateglinide",
    "chlorpropamide",
    "glimepiride",
    "acetohexamide",
    "glipizide",
    "glyburide",
    "tolbutamide",
    "pioglitazone",
    "rosiglitazone",
    "acarbose",
    "miglitol",
    "troglitazone",
    "tolazamide",
    "examide",
    "citoglipton",
    "insulin",
    "glyburide-metformin",
    "glipizide-metformin",
    "glimepiride-pioglitazone",
    "metformin-rosiglitazone",
    "metformin-pioglitazone",
)

# High-missingness columns (see design doc, "Missing Values Strategy").
# ``weight`` is ~97% missing → dropped. The others are kept and later encoded
# as an explicit "Unknown" category because their missingness is informative.
HIGH_MISSING_DROP_COLUMNS: tuple[str, ...] = ("weight",)
UNKNOWN_CATEGORY_COLUMNS: tuple[str, ...] = ("medical_specialty", "payer_code", "race")
