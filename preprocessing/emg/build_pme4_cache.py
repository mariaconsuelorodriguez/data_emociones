"""One-time extraction of features and a decimated raw-signal cache for PME4.

Reading directly from the eleven ~500MB sXX.zip archives on every training
run would be far too slow across repeated LOSO folds, so this script walks
every trial once, opens each subject's zip a single time, and writes a
single cache file with:
  - features: (N, 6*9) float32   hand-crafted features (9 per channel, 6 channels)
  - raw:      (N, 6, 5000) float32  raw EMG decimated 5kHz -> 1kHz (scipy.signal.decimate,
              which applies an anti-aliasing filter before downsampling -- a legitimate
              resampling step, not naive strided subsampling)
  - labels:   (N,) int64
  - subjects: (N,) int32

Usage:
    python preprocessing/emg/build_pme4_cache.py
"""
from __future__ import annotations

import sys
import time
import zipfile
from io import BytesIO
from pathlib import Path

import numpy as np
from scipy.signal import decimate

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from preprocessing.emg.features import extract_feature_vector
from preprocessing.emg.pme4_dataset import load_manifest

RAW_FS = 5000.0
DECIMATION_FACTOR = 5  # 5000 Hz -> 1000 Hz


def build_cache(manifest_path: Path, zip_root: Path, out_path: Path) -> None:
    trials = load_manifest(manifest_path)
    trials_by_subject: dict[int, list] = {}
    for t in trials:
        trials_by_subject.setdefault(t.subject, []).append(t)

    n = len(trials)
    features = np.zeros((n, 6 * 9), dtype=np.float32)
    raw = np.zeros((n, 6, 5000), dtype=np.float32)
    labels = np.zeros(n, dtype=np.int64)
    subjects = np.zeros(n, dtype=np.int32)

    idx = 0
    t_start = time.time()
    for subject in sorted(trials_by_subject):
        zip_path = zip_root / f"s{subject:02d}.zip"
        subject_trials = trials_by_subject[subject]
        with zipfile.ZipFile(zip_path) as zf:
            for trial in subject_trials:
                with zf.open(trial.raw_emg_filepath) as f:
                    sig = np.load(BytesIO(f.read()))  # (6, 25000) float64
                decimated = decimate(sig, DECIMATION_FACTOR, axis=-1).astype(np.float32)
                raw[idx] = decimated

                feats = np.concatenate(
                    [extract_feature_vector(sig[ch], RAW_FS) for ch in range(sig.shape[0])]
                ).astype(np.float32)
                features[idx] = feats

                labels[idx] = trial.label
                subjects[idx] = trial.subject
                idx += 1
        elapsed = time.time() - t_start
        print(f"subject {subject:02d} done ({idx}/{n} trials, {elapsed:.1f}s elapsed)")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, features=features, raw=raw, labels=labels, subjects=subjects)
    print(f"Saved cache to {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    build_cache(
        manifest_path=REPO_ROOT / "PME4" / "PME4_dataset_configs.csv",
        zip_root=REPO_ROOT / "PME4",
        out_path=REPO_ROOT / "data_cache" / "pme4" / "pme4_cache.npz",
    )
