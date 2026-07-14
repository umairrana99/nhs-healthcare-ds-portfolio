"""Tests for leakage-safe grouped cross-validation."""

from __future__ import annotations

import numpy as np

from readmission import cv


def _data() -> tuple[np.ndarray, np.ndarray]:
    # 10 patients, 3 encounters each; label is constant per patient (parity).
    groups = np.array([p for p in range(10) for _ in range(3)])
    y = np.array([p % 2 for p in range(10) for _ in range(3)])
    return y, groups


def test_make_group_splitter_config() -> None:
    splitter = cv.make_group_splitter(n_splits=4)
    assert splitter.get_n_splits() == 4


def test_group_splits_count() -> None:
    y, groups = _data()
    splits = cv.group_splits(y, groups, n_splits=5)
    assert len(splits) == 5


def test_no_patient_leaks_across_train_and_test() -> None:
    y, groups = _data()
    for train_idx, test_idx in cv.group_splits(y, groups, n_splits=5):
        train_patients = set(groups[train_idx])
        test_patients = set(groups[test_idx])
        assert train_patients.isdisjoint(test_patients)  # the whole point


def test_every_row_tested_exactly_once() -> None:
    y, groups = _data()
    tested: list[int] = []
    for _, test_idx in cv.group_splits(y, groups, n_splits=5):
        tested.extend(test_idx.tolist())
    assert sorted(tested) == list(range(len(y)))


def test_reproducible_with_seed() -> None:
    y, groups = _data()
    a = cv.group_splits(y, groups, n_splits=5, random_state=1)
    b = cv.group_splits(y, groups, n_splits=5, random_state=1)
    for (tr_a, te_a), (tr_b, te_b) in zip(a, b, strict=True):
        assert np.array_equal(tr_a, tr_b)
        assert np.array_equal(te_a, te_b)
