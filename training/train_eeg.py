"""LOSO training/evaluation for the EEG/SEED branch.

Usage:
    python training/train_eeg.py --config configs/eeg_seed.yaml

For every held-out subject (Leave-One-Subject-Out), a fresh model is
trained from scratch on the remaining subjects (with one further subject
held out for validation), evaluated on the held-out test subject, and its
embeddings are saved. All three splits are subject-disjoint, and every
normalization statistic is fit on the training fold only -- both are
checked programmatically, not just asserted in prose.
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
from models.eeg.baseline_cnn import EEGBaselineCNN
from models.eeg.cnn_bilstm import EEGCNNBiLSTM
from models.eeg.transformer import EEGTransformer
from preprocessing.common.leakage_audit import assert_disjoint_subjects, assert_scaler_fit_only_on_train
from preprocessing.eeg.seed_dataset import N_BANDS, N_CHANNELS, N_CLASSES, ZScoreNormalizer, build_sequences, load_seed, loso_splits

MODEL_BUILDERS = {
    "baseline": lambda cfg: EEGBaselineCNN(N_BANDS, N_CHANNELS, N_CLASSES, embedding_dim=cfg["model"]["embedding_dim"]),
    "cnn_bilstm": lambda cfg: EEGCNNBiLSTM(N_BANDS, N_CHANNELS, N_CLASSES, embedding_dim=cfg["model"]["embedding_dim"]),
    "transformer": lambda cfg: EEGTransformer(N_BANDS, N_CHANNELS, N_CLASSES, embedding_dim=cfg["model"]["embedding_dim"]),
}
IS_SEQUENCE_MODEL = {"baseline": False, "cnn_bilstm": True, "transformer": True}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_loader(features: np.ndarray, labels: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(torch.from_numpy(features), torch.from_numpy(labels))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def train_one_fold(model: nn.Module, train_loader, val_loader, cfg, device) -> nn.Module:
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg["training"]["learning_rate"], weight_decay=cfg["training"]["weight_decay"]
    )
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
def evaluate(model: nn.Module, loader: DataLoader, device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    if model_type not in MODEL_BUILDERS:
        raise ValueError(f"Unknown model.type '{model_type}', expected one of {list(MODEL_BUILDERS)}")

    data = load_seed(REPO_ROOT / cfg["data"]["root"])

    results_dir = REPO_ROOT / cfg["output"]["results_dir"] / model_type
    embeddings_dir = REPO_ROOT / cfg["output"]["embeddings_dir"] / model_type
    checkpoints_dir = REPO_ROOT / cfg["output"]["checkpoints_dir"] / model_type
    for d in (results_dir, embeddings_dir, checkpoints_dir):
        d.mkdir(parents=True, exist_ok=True)

    fold_metrics = []
    t_start = time.time()

    for test_subject, train_idx, val_idx, test_idx in loso_splits(data.subjects, cfg["data"]["n_val_subjects"]):
        assert_disjoint_subjects(data.subjects[train_idx], data.subjects[val_idx], data.subjects[test_idx])

        normalizer = ZScoreNormalizer().fit(data.features, train_idx)
        assert_scaler_fit_only_on_train(normalizer.fit_indices, train_idx)
        normalized = normalizer.transform(data.features)

        if IS_SEQUENCE_MODEL[model_type]:
            seq_len, stride = cfg["model"]["seq_len"], cfg["model"]["seq_stride"]
            from dataclasses import replace

            norm_data = replace(data, features=normalized)
            x_train, y_train, _ = build_sequences(norm_data, train_idx, seq_len, stride)
            x_val, y_val, _ = build_sequences(norm_data, val_idx, seq_len, stride)
            x_test, y_test, _ = build_sequences(norm_data, test_idx, seq_len, stride)
        else:
            x_train, y_train = normalized[train_idx], data.labels[train_idx]
            x_val, y_val = normalized[val_idx], data.labels[val_idx]
            x_test, y_test = normalized[test_idx], data.labels[test_idx]

        train_loader = make_loader(x_train, y_train, cfg["training"]["batch_size"], shuffle=True)
        val_loader = make_loader(x_val, y_val, cfg["training"]["batch_size"], shuffle=False)
        test_loader = make_loader(x_test, y_test, cfg["training"]["batch_size"], shuffle=False)

        model = MODEL_BUILDERS[model_type](cfg).to(device)
        model = train_one_fold(model, train_loader, val_loader, cfg, device)

        y_pred, y_true, embeddings = evaluate(model, test_loader, device)
        metrics = classification_report_dict(y_true, y_pred, N_CLASSES)
        metrics["test_subject"] = test_subject
        fold_metrics.append(metrics)

        np.savez(
            embeddings_dir / f"subject_{test_subject:02d}.npz",
            embeddings=embeddings,
            labels=y_true,
            predictions=y_pred,
        )
        torch.save(model.state_dict(), checkpoints_dir / f"subject_{test_subject:02d}.pt")

        print(
            f"[{model_type}] test_subject={test_subject:02d} "
            f"acc={metrics['accuracy']:.4f} balanced_acc={metrics['balanced_accuracy']:.4f} "
            f"f1_macro={metrics['f1_macro']:.4f} n_test={metrics['n_samples']}"
        )

    elapsed = time.time() - t_start
    aggregate = {
        "model_type": model_type,
        "n_folds": len(fold_metrics),
        "elapsed_seconds": elapsed,
        "per_fold": fold_metrics,
        "mean": {
            k: float(np.mean([m[k] for m in fold_metrics]))
            for k in ("accuracy", "balanced_accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_weighted")
        },
        "std": {
            k: float(np.std([m[k] for m in fold_metrics]))
            for k in ("accuracy", "balanced_accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_weighted")
        },
    }
    with open(results_dir / "loso_summary.json", "w") as f:
        json.dump(aggregate, f, indent=2)

    print(f"\n=== {model_type} LOSO summary over {aggregate['n_folds']} subjects ({elapsed:.1f}s) ===")
    for k, v in aggregate["mean"].items():
        print(f"  {k}: {v:.4f} +/- {aggregate['std'][k]:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "eeg_seed.yaml")
    args = parser.parse_args()
    run(args.config)
