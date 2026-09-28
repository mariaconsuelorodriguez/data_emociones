"""Builds a windowed cache for the PPGE (external-ppg) dataset.

Segments each of the 72 continuous recordings (18 subjects x 4 emotions)
into fixed-length windows, computes the cardiovascular feature vector for
each window (dropping windows with too few detected pulses rather than
fabricating HRV numbers for them), and saves everything to a single npz:
  raw:      (N, window_len) float32   z-scored per-window raw PPG
  features: (N, N_FEATURES) float32   HR/HRV/morphology features
  labels:   (N,) int64
  subjects: (N,) int32

Usage:
    python preprocessing/ppg/build_ppg_cache.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from preprocessing.ppg.external_ppg_dataset import ASSUMED_FS_HZ, load_all_recordings
from preprocessing.ppg.features import extract_window_features, segment_signal

WINDOW_SEC = 15.0
STRIDE_SEC = 5.0


def build_cache(zip_path: Path, out_path: Path) -> None:
    recordings = load_all_recordings(zip_path)
    window_len = int(WINDOW_SEC * ASSUMED_FS_HZ)

    raw_list, feat_list, label_list, subject_list = [], [], [], []
    n_dropped = 0
    for rec in recordings:
        for window in segment_signal(rec.signal, ASSUMED_FS_HZ, WINDOW_SEC, STRIDE_SEC):
            feats = extract_window_features(window, ASSUMED_FS_HZ)
            if feats is None:
                n_dropped += 1
                continue
            normalized = (window - window.mean()) / (window.std() + 1e-8)
            raw_list.append(normalized.astype(np.float32))
            feat_list.append(feats)
            label_list.append(rec.label)
            subject_list.append(rec.subject)
        print(f"subject {rec.subject:02d} {rec.emotion}: {len(rec.signal)} samples processed")

    raw = np.stack(raw_list).reshape(len(raw_list), 1, window_len)
    features = np.stack(feat_list)
    labels = np.array(label_list, dtype=np.int64)
    subjects = np.array(subject_list, dtype=np.int32)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, raw=raw, features=features, labels=labels, subjects=subjects)
    print(f"\nTotal windows kept: {len(labels)}, dropped (too few pulses detected): {n_dropped}")
    print(f"Saved cache to {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")
    print("Label distribution:", dict(zip(*np.unique(labels, return_counts=True))))
    print("Subjects present:", sorted(set(subjects.tolist())))


if __name__ == "__main__":
    build_cache(
        zip_path=REPO_ROOT / "external-ppg" / "ppg_dataset.zip",
        out_path=REPO_ROOT / "data_cache" / "ppg" / "ppg_cache.npz",
    )
