"""Leakage-safe cross-validation.

A patient (``patient_nbr``) can appear in several encounters, so a plain random
split can put the same patient in both train and test and inflate the metrics.
We use ``StratifiedGroupKFold`` grouped on the patient id: it keeps every
patient's rows within a single fold while preserving the class ratio.
"""

from __future__ import annotations

from typing import Any, TypeAlias

import numpy as np
import numpy.typing as npt
from sklearn.model_selection import StratifiedGroupKFold

DEFAULT_N_SPLITS = 5

# Accepts numpy arrays and pandas Series (pandas is untyped -> Any at call sites).
_ArrayLike: TypeAlias = npt.NDArray[Any]
_Split: TypeAlias = tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]


def make_group_splitter(
    n_splits: int = DEFAULT_N_SPLITS,
    *,
    shuffle: bool = True,
    random_state: int = 42,
) -> StratifiedGroupKFold:
    """Return a configured ``StratifiedGroupKFold``."""
    return StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=random_state,
    )


def group_splits(
    y: _ArrayLike,
    groups: _ArrayLike,
    *,
    n_splits: int = DEFAULT_N_SPLITS,
    random_state: int = 42,
) -> list[_Split]:
    """Return a list of ``(train_idx, test_idx)`` arrays grouped on ``groups``.

    ``groups`` should be the patient id per row. Stratification uses ``y``.
    """
    splitter = make_group_splitter(n_splits=n_splits, random_state=random_state)
    placeholder = np.zeros((len(y), 1))
    return [(train, test) for train, test in splitter.split(placeholder, y, groups)]
