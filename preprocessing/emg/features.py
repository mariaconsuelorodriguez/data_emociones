"""Classical hand-crafted EMG features listed in the plan (section 5/9).

These are pure signal-processing functions, independent of any dataset.
They can be validated against synthetic signals for numerical correctness
(shape/sanity only -- never as a substitute for a real experimental
result), and are the feature set for the EMG baseline (Model A: MLP on
these features).
"""
from __future__ import annotations

import numpy as np


def rms(window: np.ndarray) -> float:
    """Root Mean Square amplitude."""
    return float(np.sqrt(np.mean(np.square(window))))


def mav(window: np.ndarray) -> float:
    """Mean Absolute Value."""
    return float(np.mean(np.abs(window)))


def waveform_length(window: np.ndarray) -> float:
    """WL: cumulative length of the waveform (sum of absolute differences)."""
    return float(np.sum(np.abs(np.diff(window))))


def zero_crossings(window: np.ndarray, threshold: float = 1e-6) -> int:
    """ZC: number of sign changes, ignoring near-zero noise below `threshold`."""
    signs = np.sign(window)
    signs[np.abs(window) < threshold] = 0
    nonzero = signs[signs != 0]
    if len(nonzero) < 2:
        return 0
    return int(np.sum(np.diff(nonzero) != 0))


def slope_sign_changes(window: np.ndarray, threshold: float = 1e-6) -> int:
    """SSC: number of times the slope of the signal changes sign."""
    diffs = np.diff(window)
    signs = np.sign(diffs)
    signs[np.abs(diffs) < threshold] = 0
    nonzero = signs[signs != 0]
    if len(nonzero) < 2:
        return 0
    return int(np.sum(np.diff(nonzero) != 0))


def contraction_amplitude(window: np.ndarray) -> float:
    """Peak-to-peak amplitude of the (rectified) window."""
    rectified = np.abs(window)
    return float(np.max(rectified) - np.min(rectified))


def contraction_duration(window: np.ndarray, fs: float, threshold_ratio: float = 0.2) -> float:
    """Duration (seconds) the rectified signal stays above threshold_ratio * peak."""
    rectified = np.abs(window)
    peak = rectified.max()
    if peak == 0:
        return 0.0
    above = rectified > (threshold_ratio * peak)
    return float(np.sum(above) / fs)


def _power_spectrum(window: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    spectrum = np.abs(np.fft.rfft(window)) ** 2
    freqs = np.fft.rfftfreq(len(window), d=1.0 / fs)
    return freqs, spectrum


def mean_frequency(window: np.ndarray, fs: float) -> float:
    """MNF: power-spectrum-weighted mean frequency."""
    freqs, power = _power_spectrum(window, fs)
    total_power = power.sum()
    if total_power == 0:
        return 0.0
    return float(np.sum(freqs * power) / total_power)


def median_frequency(window: np.ndarray, fs: float) -> float:
    """MDF: frequency that splits the power spectrum in half."""
    freqs, power = _power_spectrum(window, fs)
    cumulative = np.cumsum(power)
    total_power = cumulative[-1]
    if total_power == 0:
        return 0.0
    median_idx = np.searchsorted(cumulative, total_power / 2.0)
    return float(freqs[min(median_idx, len(freqs) - 1)])


FEATURE_NAMES = [
    "rms",
    "mav",
    "waveform_length",
    "zero_crossings",
    "slope_sign_changes",
    "contraction_amplitude",
    "contraction_duration",
    "mean_frequency",
    "median_frequency",
]


def extract_feature_vector(window: np.ndarray, fs: float) -> np.ndarray:
    """Computes all 9 features for a single-channel EMG window, in FEATURE_NAMES order."""
    return np.array(
        [
            rms(window),
            mav(window),
            waveform_length(window),
            zero_crossings(window),
            slope_sign_changes(window),
            contraction_amplitude(window),
            contraction_duration(window, fs),
            mean_frequency(window, fs),
            median_frequency(window, fs),
        ],
        dtype=np.float64,
    )
