"""LOSO training/evaluation for the PPGE (external-ppg) branch.

Usage:
    python preprocessing/ppg/build_ppg_cache.py   # once, builds data_cache/ppg/ppg_cache.npz
    python training/train_ppg.py --config configs/ppg_cnn1d.yaml
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
from models.ppg.cnn1d import PPGCNN1D
from models.ppg.cnn_bilstm import PPGCNNBiLSTM
from models.ppg.transformer import PPGTransformer
from preprocessing.common.leakage_audit import assert_disjoint_subjects
from preprocessing.ppg.external_ppg_dataset import N_CLASSES


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


def build_model(model_type: str, window_len: int, embedding_dim: int) -> nn.Module:
    if model_type == "cnn1d":
        return PPGCNN1D(N_CLASSES, embedding_dim=embedding_dim)
    if model_type == "cnn_bilstm":
        return PPGCNNBiLSTM(N_CLASSES, embedding_dim=embedding_dim)
    if model_type == "transformer":
        return PPGTransformer(window_len, N_CLASSES, embedding_dim=embedding_dim)
    raise ValueError(f"Unknown model.type '{model_type}'")


def make_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def train_one_fold(model: nn.Module, train_loader, val_loader, cfg, device, class_weights) -> nn.Module:
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["training"]["learning_rate"], weight_decay=cfg["training"]["weight_decay"])
    criterion = nn.CrossEntropyLoss(weight=class_weights)
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
    raw, labels, subjects = cache["raw"], cache["labels"], cache["subjects"]
    window_len = raw.shape[-1]

    results_dir = REPO_ROOT / cfg["output"]["results_dir"] / model_type
    embeddings_dir = REPO_ROOT / cfg["output"]["embeddings_dir"] / model_type
    checkpoints_dir = REPO_ROOT / cfg["output"]["checkpoints_dir"] / model_type
    for d in (results_dir, embeddings_dir, checkpoints_dir):
        d.mkdir(parents=True, exist_ok=True)

    fold_metrics = []
    t_start = time.time()
    for test_subject, train_idx, val_idx, test_idx in loso_splits(subjects, cfg["data"]["n_val_subjects"]):
        assert_disjoint_subjects(subjects[train_idx], subjects[val_idx], subjects[test_idx])

        x_train, y_train = raw[train_idx], labels[train_idx]
        x_val, y_val = raw[val_idx], labels[val_idx]
        x_test, y_test = raw[test_idx], labels[test_idx]

        class_weights = None
        if cfg["training"]["use_class_weights"]:
            counts = np.bincount(y_train, minlength=N_CLASSES).astype(np.float32)
            class_weights = torch.tensor(counts.sum() / (N_CLASSES * counts), dtype=torch.float32)

        train_loader = make_loader(x_train, y_train, cfg["training"]["batch_size"], shuffle=True)
        val_loader = make_loader(x_val, y_val, cfg["training"]["batch_size"], shuffle=False)
        test_loader = make_loader(x_test, y_test, cfg["training"]["batch_size"], shuffle=False)

        model = build_model(model_type, window_len, cfg["model"]["embedding_dim"]).to(device)
        model = train_one_fold(model, train_loader, val_loader, cfg, device, class_weights)

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

    print(f"\n=== {model_type} PPGE LOSO summary over {aggregate['n_folds']} subjects ({elapsed:.1f}s) ===")
    for k, v in aggregate["mean"].items():
        print(f"  {k}: {v:.4f} +/- {aggregate['std'][k]:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    run(args.config)
