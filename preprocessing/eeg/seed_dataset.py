"""Loading, segmenting and splitting for the SEED EEG dataset shipped in this repo.

The three .npz files under EEG/SEED/ already contain Differential-Entropy
features (not raw EEG), verified as:
  DatasetCaricatoNoImage.npz -> arr_0, shape (50910, 5, 62) float32
    (windows x frequency bands x channels)
  LabelsNoImage.npz          -> arr_0, shape (50910,) int64, values in {0,1,2}
                                 (0=negative, 1=neutral, 2=positive per SEED convention)
  SubjectsNoImage.npz        -> arr_0, shape (50910,) int32, values in {0..14}

Samples are stored as contiguous per-subject blocks (3394 rows each), and
within a subject block as contiguous per-trial segments (label is constant
within a segment). This module never shuffles across that structure before
a split is fixed, and normalization statistics are always fit on the
training fold only.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

N_BANDS = 5
N_CHANNELS = 62
N_CLASSES = 3


@dataclass
class SeedData:
    features: np.ndarray  # (N, 5, 62) float32
    labels: np.ndarray  # (N,) int64
    subjects: np.ndarray  # (N,) int32
    segment_id: np.ndarray  # (N,) int64, unique id per contiguous (subject, trial) run


def load_seed(root: Path) -> SeedData:
    root = Path(root)
    features = np.load(root / "DatasetCaricatoNoImage.npz")["arr_0"].astype(np.float32)
    labels = np.load(root / "LabelsNoImage.npz")["arr_0"].astype(np.int64)
    subjects = np.load(root / "SubjectsNoImage.npz")["arr_0"].astype(np.int32)

    if not (features.shape[0] == labels.shape[0] == subjects.shape[0]):
        raise ValueError(
            f"Mismatched sample counts: features={features.shape[0]}, "
            f"labels={labels.shape[0]}, subjects={subjects.shape[0]}"
        )
    if features.shape[1:] != (N_BANDS, N_CHANNELS):
        raise ValueError(f"Unexpected feature shape {features.shape[1:]}, expected ({N_BANDS}, {N_CHANNELS})")

    segment_id = _contiguous_segment_ids(subjects, labels)
    return SeedData(features=features, labels=labels, subjects=subjects, segment_id=segment_id)


def _contiguous_segment_ids(subjects: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Assigns a new id every time (subject, label) changes from the previous row.

    This lets sequence models sample within a single trial without ever
    stitching together windows that actually belong to two different
    trials or two different subjects.
    """
    changed = np.zeros(len(subjects), dtype=bool)
    changed[0] = True
    changed[1:] = (subjects[1:] != subjects[:-1]) | (labels[1:] != labels[:-1])
    return np.cumsum(changed) - 1


def loso_splits(subjects: np.ndarray, n_val_subjects: int = 1) -> Iterator[tuple[int, np.ndarray, np.ndarray, np.ndarray]]:
    """Yields (test_subject_id, train_idx, val_idx, test_idx) for each subject.

    The validation fold is a different held-out subject (rotating), never
    the test subject and never mixed into train, so all three splits are
    subject-disjoint.
    """
    unique_subjects = np.sort(np.unique(subjects))
    n_subjects = len(unique_subjects)
    for i, test_subject in enumerate(unique_subjects):
        val_subjects = unique_subjects[[(i + 1 + k) % n_subjects for k in range(n_val_subjects)]]
        val_subjects = val_subjects[val_subjects != test_subject]
        train_subjects = np.array(
            [s for s in unique_subjects if s != test_subject and s not in val_subjects]
        )
        train_idx = np.where(np.isin(subjects, train_subjects))[0]
        val_idx = np.where(np.isin(subjects, val_subjects))[0]
        test_idx = np.where(subjects == test_subject)[0]
        yield int(test_subject), train_idx, val_idx, test_idx


class ZScoreNormalizer:
    """Per-band, per-channel z-score. Must be fit on the training fold only."""

    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None
        self._fit_indices: np.ndarray | None = None

    def fit(self, features: np.ndarray, indices: np.ndarray) -> "ZScoreNormalizer":
        fold = features[indices]
        self.mean_ = fold.mean(axis=0, keepdims=True)
        self.std_ = fold.std(axis=0, keepdims=True) + 1e-8
        self._fit_indices = np.array(indices)
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        if self.mean_ is None:
            raise RuntimeError("Normalizer must be fit before transform.")
        return (features - self.mean_) / self.std_

    @property
    def fit_indices(self) -> np.ndarray:
        assert self._fit_indices is not None
        return self._fit_indices


def build_sequences(
    data: SeedData, indices: np.ndarray, seq_len: int, stride: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Builds fixed-length sequences of consecutive windows within a single segment.

    Returns (sequences, labels, subjects) where sequences has shape
    (n_sequences, seq_len, 5, 62). A sequence is only formed from indices
    that share the same segment_id, so it never crosses a trial or
    subject boundary.
    """
    indices = np.sort(indices)
    seqs, seq_labels, seq_subjects = [], [], []
    for seg in np.unique(data.segment_id[indices]):
        seg_idx = indices[data.segment_id[indices] == seg]
        if len(seg_idx) < seq_len:
            continue
        for start in range(0, len(seg_idx) - seq_len + 1, stride):
            window = seg_idx[start : start + seq_len]
            seqs.append(data.features[window])
            seq_labels.append(data.labels[window[-1]])
            seq_subjects.append(data.subjects[window[-1]])
    if not seqs:
        return (
            np.empty((0, seq_len, N_BANDS, N_CHANNELS), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.int32),
        )
    return np.stack(seqs), np.array(seq_labels, dtype=np.int64), np.array(seq_subjects, dtype=np.int32)
