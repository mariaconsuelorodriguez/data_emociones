"""Loader for the real PPGE dataset (external-ppg/), 18 subjects x 4 emotions.

Verified real content: external-ppg/ppg_dataset.zip contains 72 files named
"<subject>/<emotion>_<subject>.txt", one raw PPG sample per line (plain
integers, plausible ADC counts from a low-cost sensor), one continuous
recording per subject/emotion (length varies with each video clip's
duration, ~20,000-30,000 samples).

IMPORTANT: the source repository does not document a sampling rate
anywhere (checked its README and GitHub page). ASSUMED_FS_HZ below is a
value the user supplied directly in this session, not one published by
the dataset's authors -- any HR/HRV feature expressed in real seconds/bpm
inherits that uncertainty and should be reported with this caveat.
"""
from __future__ import annotations

import re
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ASSUMED_FS_HZ = 100.0  # user-supplied; not documented by the dataset's authors

EMOTION_TO_LABEL = {"anger": 0, "joy": 1, "relaxed": 2, "sadness": 3}
N_CLASSES = len(EMOTION_TO_LABEL)
N_SUBJECTS = 18

_FILENAME_RE = re.compile(r"(\d+)/(\w+)_(\d+)\.txt")


@dataclass
class PpgRecording:
    subject: int
    label: int
    emotion: str
    signal: np.ndarray  # 1D raw PPG, length varies


def load_all_recordings(zip_path: Path) -> list[PpgRecording]:
    recordings = []
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            m = _FILENAME_RE.match(name)
            if not m:
                continue
            subject, emotion, subject2 = int(m.group(1)), m.group(2), int(m.group(3))
            if subject != subject2:
                raise ValueError(f"Inconsistent subject id in filename: {name}")
            if emotion not in EMOTION_TO_LABEL:
                raise ValueError(f"Unexpected emotion '{emotion}' in {name}")
            with zf.open(name) as f:
                signal = np.array([float(line) for line in f.read().decode().splitlines() if line.strip()], dtype=np.float32)
            recordings.append(PpgRecording(subject=subject, label=EMOTION_TO_LABEL[emotion], emotion=emotion, signal=signal))
    if len(recordings) != N_SUBJECTS * N_CLASSES:
        raise ValueError(f"Expected {N_SUBJECTS * N_CLASSES} recordings, found {len(recordings)}")
    return recordings
