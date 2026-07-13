"""Helpers for obtaining the UCI Diabetes 130-US Hospitals dataset.

The raw data is licensed CC BY 4.0 but is intentionally NOT committed to the
repository (see the root ``.gitignore``). Fetch it locally with
:func:`download_dataset` or follow :func:`get_download_instructions`.
"""

from __future__ import annotations

import io
import urllib.request
import zipfile
from pathlib import Path

from readmission.logging_utils import get_logger

logger = get_logger(__name__)

DATASET_URL = (
    "https://archive.ics.uci.edu/static/public/296/"
    "diabetes+130-us+hospitals+for+years+1999-2008.zip"
)
# Path of the main CSV inside the distributed zip archive.
CSV_MEMBER = "dataset_diabetes/diabetic_data.csv"


def get_download_instructions(raw_dir: Path) -> str:
    """Return human-readable instructions for obtaining the dataset."""
    return (
        "UCI Diabetes 130-US Hospitals (CC BY 4.0)\n"
        f"  1. Download: {DATASET_URL}\n"
        f"  2. Unzip and place 'diabetic_data.csv' in: {raw_dir}\n"
        "  Or run: python -m readmission.data.download\n"
        "Source: https://archive.ics.uci.edu/dataset/296/"
        "diabetes+130-us+hospitals+for+years+1999-2008"
    )


def download_dataset(raw_dir: Path) -> Path:
    """Download and extract ``diabetic_data.csv`` into ``raw_dir``.

    Returns the path to the extracted CSV. Requires network access.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    target = raw_dir / "diabetic_data.csv"
    if target.exists():
        logger.info("Dataset already present at %s", target)
        return target

    logger.info("Downloading dataset from %s", DATASET_URL)
    with urllib.request.urlopen(DATASET_URL) as response:  # noqa: S310 (trusted UCI host)
        archive_bytes = response.read()

    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        with archive.open(CSV_MEMBER) as member, target.open("wb") as out:
            out.write(member.read())
    logger.info("Extracted %s", target)
    return target


if __name__ == "__main__":
    from readmission.config import get_settings
    from readmission.logging_utils import configure_logging

    configure_logging()
    settings = get_settings()
    download_dataset(settings.raw_data_path.parent)
