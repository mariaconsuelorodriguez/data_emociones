"""RAF-DB (7 basic emotions, single-label) dataset reader.

Verified real content in this repo, under RAF-DB/:
  RAF-DB/DATASET/train/<1-7>/train_XXXXX_aligned.jpg  (12,271 images)
  RAF-DB/DATASET/test/<1-7>/test_XXXX_aligned.jpg      (3,068 images)
  RAF-DB/train_labels.csv, RAF-DB/test_labels.csv       (image,label CSVs)

The per-class folder names (1-7) and the CSV label column agree exactly
with the class distribution of the official RAF-DB benchmark (label 4 =
Happy is the majority class), so this loader reads directly from the
folder structure -- one class per subfolder -- rather than needing to
join against the CSV. The CSVs are kept for reference/cross-checking.
Labels use RAF-DB's original 1-7 encoding; this module remaps them to
0-6 for a zero-indexed classifier.
"""
from __future__ import annotations

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
    image_path: Path
    label: int  # 0-6, zero-indexed


def _scan_split(dataset_root: Path, split: str) -> list[RafDbSample]:
    split_dir = Path(dataset_root) / split
    if not split_dir.exists():
        raise FileNotFoundError(
            f"RAF-DB split directory not found: {split_dir}. Expected "
            f"RAF-DB/DATASET/{split}/<1-7>/<image>.jpg."
        )
    samples = []
    for raw_label in sorted(RAF_DB_LABEL_NAMES):
        class_dir = split_dir / str(raw_label)
        if not class_dir.exists():
            raise FileNotFoundError(f"Missing RAF-DB class folder: {class_dir}")
        for image_path in sorted(class_dir.glob("*.jpg")):
            samples.append(RafDbSample(image_path=image_path, label=raw_label - 1))
    if not samples:
        raise FileNotFoundError(f"No images found under {split_dir}")
    return samples


class RafDbDataset(Dataset):
    """dataset_root should point at RAF-DB/DATASET; split is "train" or "test"."""

    def __init__(self, dataset_root: Path, split: str, transform=None):
        if split not in ("train", "test"):
            raise ValueError(f"split must be 'train' or 'test', got {split!r}")
        self.samples = _scan_split(Path(dataset_root), split)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        from PIL import Image

        sample = self.samples[idx]
        image = Image.open(sample.image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, sample.label
