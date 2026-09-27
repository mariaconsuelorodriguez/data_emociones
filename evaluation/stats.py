"""Statistical comparison between per-subject metrics of two or more models.

Follows the plan's rule: never apply ANOVA by default. Normality is tested
first (Shapiro-Wilk on the paired differences for two models, or on each
group for more than two), and the parametric vs. non-parametric route is
chosen from that result and reported explicitly.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class ComparisonResult:
    test_name: str
    statistic: float
    p_value: float
    normality_p_value: float
    used_parametric: bool
    justification: str


def compare_two_models(metric_a: np.ndarray, metric_b: np.ndarray, alpha: float = 0.05) -> ComparisonResult:
    """Paired comparison of the same per-subject metric between two models (e.g. LOSO folds)."""
    if len(metric_a) != len(metric_b):
        raise ValueError("Both metric arrays must have one value per subject/fold, in matching order.")
    diff = metric_a - metric_b
    _, normality_p = stats.shapiro(diff)
    if normality_p > alpha:
        stat, p = stats.ttest_rel(metric_a, metric_b)
        return ComparisonResult(
            test_name="paired t-test",
            statistic=float(stat),
            p_value=float(p),
            normality_p_value=float(normality_p),
            used_parametric=True,
            justification=(
                f"Shapiro-Wilk on paired differences p={normality_p:.4f} > {alpha}: "
                "normality not rejected, so a paired t-test is used."
            ),
        )
    stat, p = stats.wilcoxon(metric_a, metric_b)
    return ComparisonResult(
        test_name="Wilcoxon signed-rank",
        statistic=float(stat),
        p_value=float(p),
        normality_p_value=float(normality_p),
        used_parametric=False,
        justification=(
            f"Shapiro-Wilk on paired differences p={normality_p:.4f} <= {alpha}: "
            "normality rejected, so the non-parametric Wilcoxon signed-rank test is used instead of a t-test."
        ),
    )


def compare_many_models(metrics_by_model: dict[str, np.ndarray], alpha: float = 0.05) -> ComparisonResult:
    """Compares 3+ models (e.g. A/B/C) evaluated on the same per-subject folds."""
    arrays = list(metrics_by_model.values())
    n = len(arrays[0])
    if any(len(a) != n for a in arrays):
        raise ValueError("All models must have the same number of per-subject values.")
    normality_ps = [stats.shapiro(a)[1] for a in arrays]
    worst_normality_p = min(normality_ps)
    if worst_normality_p > alpha:
        stat, p = stats.f_oneway(*arrays)
        return ComparisonResult(
            test_name="repeated-measures ANOVA (one-way approximation)",
            statistic=float(stat),
            p_value=float(p),
            normality_p_value=float(worst_normality_p),
            used_parametric=True,
            justification=(
                f"Shapiro-Wilk per model: min p={worst_normality_p:.4f} > {alpha}: "
                "normality not rejected for any model, so ANOVA is used."
            ),
        )
    stat, p = stats.friedmanchisquare(*arrays)
    return ComparisonResult(
        test_name="Friedman",
        statistic=float(stat),
        p_value=float(p),
        normality_p_value=float(worst_normality_p),
        used_parametric=False,
        justification=(
            f"Shapiro-Wilk per model: min p={worst_normality_p:.4f} <= {alpha}: "
            "normality rejected for at least one model, so the non-parametric Friedman test is used instead of ANOVA "
            "(follow up with pairwise Wilcoxon + correction if this is significant)."
        ),
    )
