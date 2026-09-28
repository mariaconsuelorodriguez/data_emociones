"""PPG preprocessing and cardiovascular feature extraction (plan section 6/9).

Uses NeuroKit2 for cleaning and pulse detection (as recommended in the
plan's reproducibility section) plus a small set of hand-crafted
time/frequency features: HR, PRV/HRV (SDNN, RMSSD, pNN50), pulse
amplitude/morphology, inter-pulse intervals, and a dominant-frequency
time-frequency feature.

Sanity-checked against a real recording at the user-supplied 100 Hz
sampling rate: peak detection over a 20s window gives ~75 bpm with
inter-beat intervals of ~0.8-0.9s, both physiologically plausible for a
resting adult -- consistent with (though not proof of) the assumed rate.
"""
from __future__ import annotations

import numpy as np
import neurokit2 as nk
from scipy.signal import welch

FEATURE_NAMES = [
    "hr_mean_bpm",
    "hr_std_bpm",
    "sdnn_ms",
    "rmssd_ms",
    "pnn50",
    "amplitude_mean",
    "amplitude_std",
    "ibi_mean_s",
    "ibi_std_s",
    "dominant_freq_hz",
    "n_peaks",
]
N_FEATURES = len(FEATURE_NAMES)
MIN_PEAKS_FOR_HRV = 4


def extract_window_features(window: np.ndarray, fs: float) -> np.ndarray | None:
    """Returns an (N_FEATURES,) vector, or None if too few pulses were
    detected in this window to compute reliable HRV statistics (the window
    should then be dropped rather than filled with fabricated numbers)."""
    try:
        cleaned = nk.ppg_clean(window, sampling_rate=int(fs))
        peaks_info = nk.ppg_findpeaks(cleaned, sampling_rate=int(fs))
        peaks = peaks_info["PPG_Peaks"]
    except Exception:
        return None

    if len(peaks) < MIN_PEAKS_FOR_HRV:
        return None

    ibi_s = np.diff(peaks) / fs  # inter-beat intervals in seconds
    hr_bpm = 60.0 / ibi_s

    sdnn_ms = ibi_s.std(ddof=1) * 1000.0 if len(ibi_s) > 1 else 0.0
    diffs_ms = np.diff(ibi_s) * 1000.0
    rmssd_ms = float(np.sqrt(np.mean(diffs_ms**2))) if len(diffs_ms) > 0 else 0.0
    pnn50 = float(np.mean(np.abs(diffs_ms) > 50.0)) if len(diffs_ms) > 0 else 0.0

    # Morphology: amplitude of each detected pulse (peak-to-preceding-trough proxy via cleaned signal range around each peak)
    amplitudes = []
    half_win = int(0.3 * fs)
    for p in peaks:
        lo, hi = max(0, p - half_win), min(len(cleaned), p + half_win)
        amplitudes.append(cleaned[lo:hi].max() - cleaned[lo:hi].min())
    amplitudes = np.array(amplitudes)

    freqs, psd = welch(cleaned, fs=fs, nperseg=min(len(cleaned), int(fs * 8)))
    band = (freqs >= 0.5) & (freqs <= 3.0)  # plausible HR band, 30-180 bpm
    dominant_freq = float(freqs[band][np.argmax(psd[band])]) if band.any() else 0.0

    return np.array(
        [
            hr_bpm.mean(),
            hr_bpm.std() if len(hr_bpm) > 1 else 0.0,
            sdnn_ms,
            rmssd_ms,
            pnn50,
            amplitudes.mean(),
            amplitudes.std() if len(amplitudes) > 1 else 0.0,
            ibi_s.mean(),
            ibi_s.std() if len(ibi_s) > 1 else 0.0,
            dominant_freq,
            float(len(peaks)),
        ],
        dtype=np.float32,
    )


def segment_signal(signal: np.ndarray, fs: float, window_sec: float, stride_sec: float) -> list[np.ndarray]:
    """Non-overlapping-by-default sliding windows over one continuous recording.
    Never crosses into a different recording (caller passes one recording at a time)."""
    window_len = int(window_sec * fs)
    stride_len = int(stride_sec * fs)
    windows = []
    for start in range(0, len(signal) - window_len + 1, stride_len):
        windows.append(signal[start : start + window_len])
    return windows
