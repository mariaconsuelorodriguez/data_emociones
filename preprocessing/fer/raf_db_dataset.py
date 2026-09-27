"""RAF-DB (7 basic emotions, single-label) dataset reader.

train_labels.csv and test_labels.csv are already in this repo and were
verified against the known RAF-DB benchmark: 12,271 train / 3,068 test
rows, class distribution matching the official split (label 4 = Happy is
the majority class). Labels use RAF-DB's original 1-7 encoding; this
module remaps them to 0-6 for use with a zero-indexed classifier.

The actual images (train_00001_aligned.jpg, ...) are NOT in the repo yet
-- only the label CSVs. This loader fails loudly and specifically (which
image is missing, and where it looked) instead of silently fabricating
data, per the project's no-simulated-results rule.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from torch.utils.data import Dataset

RAF_DB_LABEL_NAMES = {
    1: "surprise",
    2: "fear",
    3: "disgust",
    4: "happy",
    5: "sad",
    6: "anger",
    7: "neutral",
}
N_CLASSES = 7


@dataclass
class RafDbSample:
    image_name: str
    label: int  # 0-6, zero-indexed


def _read_labels(csv_path: Path) -> list[RafDbSample]:
    samples = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_label = int(row["label"])
            if raw_label not in RAF_DB_LABEL_NAMES:
                raise ValueError(f"Unexpected RAF-DB label {raw_label} in {csv_path} (expected 1-7)")
            samples.append(RafDbSample(image_name=row["image"], label=raw_label - 1))
    return samples


class RafDbDataset(Dataset):
    """Expects images under `image_dir/<image_name>`. Raises FileNotFoundError
    with the exact missing path the first time an image can't be found, so a
    missing upload is diagnosed immediately instead of producing empty/blank
    tensors."""

    def __init__(self, labels_csv: Path, image_dir: Path, transform=None):
        self.samples = _read_labels(Path(labels_csv))
        self.image_dir = Path(image_dir)
        self.transform = transform
        if not self.image_dir.exists():
            raise FileNotFoundError(
                f"RAF-DB image directory not found: {self.image_dir}. "
                "Only the label CSVs are in this repo so far; upload the aligned "
                "images (e.g. train_00001_aligned.jpg) under this path before training."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        from PIL import Image

        sample = self.samples[idx]
        image_path = self.image_dir / sample.image_name
        if not image_path.exists():
            raise FileNotFoundError(f"Missing RAF-DB image: {image_path}")
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, sample.label
