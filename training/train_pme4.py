"""LOSO training/evaluation for the PME4 EMG branch.

Usage:
    python preprocessing/emg/build_pme4_cache.py   # once, builds data_cache/pme4/pme4_cache.npz
    python training/train_pme4.py --config configs/pme4_mlp.yaml

Uses the cached feature/raw-signal arrays (see build_pme4_cache.py) so that
LOSO over 11 subjects doesn't have to re-read the ~500MB-per-subject zip
archives on every fold. Subject-disjoint LOSO splits, leakage-audited the
same way as the EEG branch.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from evaluation.metrics import classification_report_dict
from models.emg.cnn1d import EMGCNN1D
from models.emg.cnn_lstm import EMGCNNLSTM
from models.emg.mlp_features import EMGBaselineMLP
from preprocessing.common.leakage_audit import assert_disjoint_subjects, assert_scaler_fit_only_on_train
from preprocessing.emg.pme4_dataset import N_CLASSES

N_CHANNELS = 6
N_FEATURES = 6 * 9


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def loso_splits(subjects: np.ndarray, n_val_subjects: int = 1):
    unique_subjects = np.sort(np.unique(subjects))
    n = len(unique_subjects)
    for i, test_subject in enumerate(unique_subjects):
        val_subjects = unique_subjects[[(i + 1 + k) % n for k in range(n_val_subjects)]]
        val_subjects = val_subjects[val_subjects != test_subject]
        train_subjects = np.array([s for s in unique_subjects if s != test_subject and s not in val_subjects])
        train_idx = np.where(np.isin(subjects, train_subjects))[0]
        val_idx = np.where(np.isin(subjects, val_subjects))[0]
        test_idx = np.where(subjects == test_subject)[0]
        yield int(test_subject), train_idx, val_idx, test_idx


class ZScoreNormalizer:
    def __init__(self):
        self.mean_ = None
        self.std_ = None
        self._fit_indices = None

    def fit(self, x: np.ndarray, indices: np.ndarray) -> "ZScoreNormalizer":
        fold = x[indices]
        self.mean_ = fold.mean(axis=0, keepdims=True)
        self.std_ = fold.std(axis=0, keepdims=True) + 1e-8
        self._fit_indices = np.array(indices)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean_) / self.std_

    @property
    def fit_indices(self) -> np.ndarray:
        return self._fit_indices


def build_model(model_type: str, embedding_dim: int) -> nn.Module:
    if model_type == "mlp_features":
        return EMGBaselineMLP(N_FEATURES, N_CLASSES, embedding_dim=embedding_dim)
    if model_type == "cnn1d":
        return EMGCNN1D(N_CHANNELS, N_CLASSES, embedding_dim=embedding_dim)
    if model_type == "cnn_lstm":
        return EMGCNNLSTM(N_CHANNELS, N_CLASSES, embedding_dim=embedding_dim)
    raise ValueError(f"Unknown model.type '{model_type}'")


def make_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def train_one_fold(model: nn.Module, train_loader, val_loader, cfg, device) -> nn.Module:
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["training"]["learning_rate"], weight_decay=cfg["training"]["weight_decay"])
    criterion = nn.CrossEntropyLoss()
    best_val_acc, best_state = -1.0, None
    for _epoch in range(cfg["training"]["epochs"]):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits, _ = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits, _ = model(xb)
                pred = logits.argmax(dim=-1)
                correct += (pred == yb).sum().item()
                total += yb.numel()
        val_acc = correct / max(total, 1)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device):
    model.eval()
    all_pred, all_true, all_embed = [], [], []
    for xb, yb in loader:
        xb = xb.to(device)
        logits, embedding = model(xb)
        all_pred.append(logits.argmax(dim=-1).cpu().numpy())
        all_true.append(yb.numpy())
        all_embed.append(embedding.cpu().numpy())
    return np.concatenate(all_pred), np.concatenate(all_true), np.concatenate(all_embed)


def run(config_path: Path) -> None:
    cfg = yaml.safe_load(config_path.read_text())
    set_seed(cfg["seed"])
    device = torch.device(cfg["training"]["device"])
    model_type = cfg["model"]["type"]

    cache = np.load(REPO_ROOT / cfg["data"]["cache_path"])
    labels, subjects = cache["labels"], cache["subjects"]
    if model_type == "mlp_features":
        x_all = cache["features"]
    else:
        raw = cache["raw"]  # (N, 6, 5000)
        x_all = raw if model_type != "mlp_features" else raw.reshape(len(raw), -1)

    results_dir = REPO_ROOT / cfg["output"]["results_dir"] / model_type
    embeddings_dir = REPO_ROOT / cfg["output"]["embeddings_dir"] / model_type
    checkpoints_dir = REPO_ROOT / cfg["output"]["checkpoints_dir"] / model_type
    for d in (results_dir, embeddings_dir, checkpoints_dir):
        d.mkdir(parents=True, exist_ok=True)

    fold_metrics = []
    t_start = time.time()
    for test_subject, train_idx, val_idx, test_idx in loso_splits(subjects, cfg["data"]["n_val_subjects"]):
        assert_disjoint_subjects(subjects[train_idx], subjects[val_idx], subjects[test_idx])

        # Normalize per-feature (mlp) or per-channel (raw) using train fold only.
        if model_type == "mlp_features":
            normalizer = ZScoreNormalizer().fit(x_all, train_idx)
        else:
            flat = x_all.reshape(len(x_all), -1)
            normalizer = ZScoreNormalizer().fit(flat, train_idx)
        assert_scaler_fit_only_on_train(normalizer.fit_indices, train_idx)

        if model_type == "mlp_features":
            x_norm = normalizer.transform(x_all)
        else:
            x_norm = normalizer.transform(x_all.reshape(len(x_all), -1)).reshape(x_all.shape).astype(np.float32)

        x_train, y_train = x_norm[train_idx], labels[train_idx]
        x_val, y_val = x_norm[val_idx], labels[val_idx]
        x_test, y_test = x_norm[test_idx], labels[test_idx]

        train_loader = make_loader(x_train, y_train, cfg["training"]["batch_size"], shuffle=True)
        val_loader = make_loader(x_val, y_val, cfg["training"]["batch_size"], shuffle=False)
        test_loader = make_loader(x_test, y_test, cfg["training"]["batch_size"], shuffle=False)

        model = build_model(model_type, cfg["model"]["embedding_dim"]).to(device)
        model = train_one_fold(model, train_loader, val_loader, cfg, device)

        y_pred, y_true, embeddings = evaluate(model, test_loader, device)
        metrics = classification_report_dict(y_true, y_pred, N_CLASSES)
        metrics["test_subject"] = test_subject
        fold_metrics.append(metrics)

        np.savez(embeddings_dir / f"subject_{test_subject:02d}.npz", embeddings=embeddings, labels=y_true, predictions=y_pred)
        torch.save(model.state_dict(), checkpoints_dir / f"subject_{test_subject:02d}.pt")

        print(
            f"[{model_type}] test_subject={test_subject:02d} acc={metrics['accuracy']:.4f} "
            f"balanced_acc={metrics['balanced_accuracy']:.4f} f1_macro={metrics['f1_macro']:.4f} n_test={metrics['n_samples']}"
        )

    elapsed = time.time() - t_start
    aggregate = {
        "model_type": model_type,
        "n_folds": len(fold_metrics),
        "elapsed_seconds": elapsed,
        "per_fold": fold_metrics,
        "mean": {k: float(np.mean([m[k] for m in fold_metrics])) for k in ("accuracy", "balanced_accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_weighted")},
        "std": {k: float(np.std([m[k] for m in fold_metrics])) for k in ("accuracy", "balanced_accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_weighted")},
    }
    with open(results_dir / "loso_summary.json", "w") as f:
        json.dump(aggregate, f, indent=2)

    print(f"\n=== {model_type} PME4 LOSO summary over {aggregate['n_folds']} subjects ({elapsed:.1f}s) ===")
    for k, v in aggregate["mean"].items():
        print(f"  {k}: {v:.4f} +/- {aggregate['std'][k]:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    run(args.config)
