"""Cross-cutting leakage checks shared by every modality's LOSO pipeline."""
from __future__ import annotations

import numpy as np


class LeakageError(RuntimeError):
    pass


def assert_disjoint_subjects(train_subjects: np.ndarray, val_subjects: np.ndarray, test_subjects: np.ndarray) -> None:
    train_set = set(np.unique(train_subjects).tolist())
    val_set = set(np.unique(val_subjects).tolist())
    test_set = set(np.unique(test_subjects).tolist())
    overlap_tv = train_set & val_set
    overlap_tt = train_set & test_set
    overlap_vt = val_set & test_set
    if overlap_tv or overlap_tt or overlap_vt:
        raise LeakageError(
            f"Subject leakage detected: train/val={overlap_tv}, train/test={overlap_tt}, val/test={overlap_vt}"
        )


def assert_scaler_fit_only_on_train(scaler_fit_indices: np.ndarray, train_indices: np.ndarray) -> None:
    if not np.array_equal(np.sort(scaler_fit_indices), np.sort(train_indices)):
        raise LeakageError("Normalization statistics were fit on indices outside the training fold.")
