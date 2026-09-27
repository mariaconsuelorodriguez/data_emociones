"""PME4 manifest reader (EMG branch).

PME4/PME4_dataset_configs.csv is a real, verified manifest: 3,829 trials
across 11 subjects and 7 balanced emotion classes (anger/disgust/fear/
happy/neutral/sad/surprise), with per-trial paths to raw EEG, raw EMG,
audio and face-feature files. Only the manifest is in this repo so far --
the actual .npy/.wav files it points to are not, so this module parses
the manifest and exposes a subject-based LOSO split, but raises a clear
error naming the missing file instead of fabricating signal data.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

EMOTION_TO_LABEL = {
    "anger": 0,
    "disgust": 1,
    "fear": 2,
    "happy": 3,
    "neutral": 4,
    "sad": 5,
    "surprise": 6,
}
N_CLASSES = len(EMOTION_TO_LABEL)


@dataclass
class Pme4Trial:
    subject: int
    trial: int
    label: int
    emotion: str
    raw_emg_filepath: str
    processed_emg_filepath: str
    raw_eeg_filepath: str
    processed_eeg_filepath: str
    audio_wav_filepath: str
    face_vgg16_features_filepath: str


def load_manifest(csv_path: Path) -> list[Pme4Trial]:
    trials = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            emotion = row["emotion"].strip()
            if emotion not in EMOTION_TO_LABEL:
                raise ValueError(f"Unexpected PME4 emotion '{emotion}' in {csv_path}")
            trials.append(
                Pme4Trial(
                    subject=int(row["subject"]),
                    trial=int(row["trial"]),
                    label=EMOTION_TO_LABEL[emotion],
                    emotion=emotion,
                    raw_emg_filepath=row["raw_emg_filepath"],
                    processed_emg_filepath=row["processed_emg_filepath"],
                    raw_eeg_filepath=row["raw_eeg_filepath"],
                    processed_eeg_filepath=row["processed_eeg_filepath"],
                    audio_wav_filepath=row["audio_wav_filepath"],
                    face_vgg16_features_filepath=row.get("face_vgg16_features_filepath", ""),
                )
            )
    return trials


def loso_splits(trials: list[Pme4Trial], n_val_subjects: int = 1) -> Iterator[tuple[int, list[Pme4Trial], list[Pme4Trial], list[Pme4Trial]]]:
    subjects = sorted({t.subject for t in trials})
    n = len(subjects)
    for i, test_subject in enumerate(subjects):
        val_subjects = {subjects[(i + 1 + k) % n] for k in range(n_val_subjects)} - {test_subject}
        train = [t for t in trials if t.subject != test_subject and t.subject not in val_subjects]
        val = [t for t in trials if t.subject in val_subjects]
        test = [t for t in trials if t.subject == test_subject]
        yield test_subject, train, val, test


def load_raw_emg(trial: Pme4Trial, data_root: Path) -> np.ndarray:
    """Loads the raw EMG signal for one trial. Raises FileNotFoundError with the
    exact missing path if the referenced .npy file hasn't been uploaded yet."""
    path = Path(data_root) / trial.raw_emg_filepath
    if not path.exists():
        raise FileNotFoundError(
            f"Missing PME4 raw EMG file: {path}. The manifest is in the repo but the "
            "underlying signal files still need to be uploaded (see PLAN_EXPERIMENTAL.md, section 7)."
        )
    return np.load(path)
