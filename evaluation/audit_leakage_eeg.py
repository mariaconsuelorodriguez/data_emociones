"""Standalone leakage audit for the EEG/SEED LOSO splits (plan section 16).

Checks, for every one of the 15 folds:
  - train/val/test subjects are pairwise disjoint;
  - the normalization scaler's fit indices exactly match the training fold
    (never touching validation or test rows);
  - no sample index appears in more than one split.

Writes results/eeg/seed/leakage_audit.json. Exits non-zero if any check fails.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from preprocessing.common.leakage_audit import LeakageError, assert_disjoint_subjects, assert_scaler_fit_only_on_train
from preprocessing.eeg.seed_dataset import ZScoreNormalizer, load_seed, loso_splits


def main() -> None:
    data = load_seed(REPO_ROOT / "EEG" / "SEED")
    report = {"dataset": "EEG/SEED", "n_samples": int(len(data.subjects)), "folds": []}
    all_ok = True

    for test_subject, train_idx, val_idx, test_idx in loso_splits(data.subjects, n_val_subjects=1):
        fold_report = {"test_subject": test_subject, "n_train": len(train_idx), "n_val": len(val_idx), "n_test": len(test_idx)}
        try:
            assert_disjoint_subjects(data.subjects[train_idx], data.subjects[val_idx], data.subjects[test_idx])

            all_idx = np.concatenate([train_idx, val_idx, test_idx])
            if len(all_idx) != len(np.unique(all_idx)):
                raise LeakageError("A sample index appears in more than one split.")

            normalizer = ZScoreNormalizer().fit(data.features, train_idx)
            assert_scaler_fit_only_on_train(normalizer.fit_indices, train_idx)

            fold_report["status"] = "OK"
        except LeakageError as e:
            fold_report["status"] = f"FAIL: {e}"
            all_ok = False
        report["folds"].append(fold_report)

    report["all_folds_passed"] = all_ok

    out_path = REPO_ROOT / "results" / "eeg" / "seed" / "leakage_audit.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))

    print(f"Leakage audit: {'ALL PASSED' if all_ok else 'FAILURES FOUND'} -- see {out_path}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
