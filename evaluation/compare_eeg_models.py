"""Statistical comparison of the EEG/SEED A/B/C LOSO results (plan section 11).

Loads the three per-fold LOSO summaries and compares their per-subject
accuracy with compare_many_models (which picks ANOVA vs. Friedman based on
a Shapiro-Wilk normality check -- never ANOVA by default), plus the three
pairwise comparisons.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from evaluation.stats import compare_many_models, compare_two_models

MODELS = {
    "A_baseline": REPO_ROOT / "results/eeg/seed/baseline/loso_summary.json",
    "B_cnn_bilstm": REPO_ROOT / "results/eeg/seed/cnn_bilstm/loso_summary.json",
    "C_transformer": REPO_ROOT / "results/eeg/seed/transformer/loso_summary.json",
}


def per_subject_metric(summary_path: Path, metric: str) -> np.ndarray:
    summary = json.loads(summary_path.read_text())
    by_subject = {fold["test_subject"]: fold[metric] for fold in summary["per_fold"]}
    return np.array([by_subject[s] for s in sorted(by_subject)])


def main() -> None:
    metric = "accuracy"
    per_subject = {name: per_subject_metric(path, metric) for name, path in MODELS.items()}

    print(f"=== Per-subject '{metric}' (15 subjects, LOSO) ===")
    for name, values in per_subject.items():
        print(f"  {name}: mean={values.mean():.4f} std={values.std():.4f}")

    print("\n=== Omnibus comparison (A vs B vs C) ===")
    result = compare_many_models(per_subject)
    print(f"  test: {result.test_name}")
    print(f"  statistic={result.statistic:.4f}, p={result.p_value:.4f}")
    print(f"  {result.justification}")

    print("\n=== Pairwise comparisons ===")
    names = list(per_subject.keys())
    output = {"metric": metric, "per_subject": {k: v.tolist() for k, v in per_subject.items()}, "omnibus": vars(result), "pairwise": {}}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            pair_result = compare_two_models(per_subject[a], per_subject[b])
            print(f"  {a} vs {b}: {pair_result.test_name}, statistic={pair_result.statistic:.4f}, p={pair_result.p_value:.4f}")
            print(f"    {pair_result.justification}")
            output["pairwise"][f"{a}_vs_{b}"] = vars(pair_result)

    out_path = REPO_ROOT / "results/eeg/seed/model_comparison.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
