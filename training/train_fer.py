"""Training/evaluation for the RAF-DB FER branch.

Usage:
    python training/train_fer.py --config configs/fer_raf_db.yaml

RAF-DB has no subject IDs (in-the-wild web images), so this uses the
official train/test split plus a stratified validation split carved out
of train -- not LOSO, per PLAN_EXPERIMENTAL.md section 3.
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
import torchvision.transforms as T
import yaml
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from evaluation.metrics import classification_report_dict
from models.fer.baseline_cnn import FERBaselineCNN
from models.fer.resnet_advanced import FERResNet18
from preprocessing.fer.raf_db_dataset import N_CLASSES, RafDbDataset


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def stratified_split(labels: list[int], val_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    labels = np.array(labels)
    train_idx, val_idx = [], []
    for c in np.unique(labels):
        class_idx = np.where(labels == c)[0]
        rng.shuffle(class_idx)
        n_val = max(1, int(len(class_idx) * val_fraction))
        val_idx.extend(class_idx[:n_val])
        train_idx.extend(class_idx[n_val:])
    return np.array(sorted(train_idx)), np.array(sorted(val_idx))


def build_model(cfg: dict) -> nn.Module:
    model_type = cfg["model"]["type"]
    if model_type == "baseline":
        return FERBaselineCNN(N_CLASSES, embedding_dim=cfg["model"]["embedding_dim"])
    if model_type == "resnet18":
        try:
            return FERResNet18(N_CLASSES, embedding_dim=cfg["model"]["embedding_dim"], pretrained=cfg["model"]["pretrained"])
        except Exception as e:  # e.g. blocked network access to download pretrained weights
            print(f"WARNING: could not load pretrained ResNet18 weights ({e}); falling back to random init.")
            return FERResNet18(N_CLASSES, embedding_dim=cfg["model"]["embedding_dim"], pretrained=False)
    raise ValueError(f"Unknown model.type '{model_type}'")


def train_and_evaluate(cfg: dict) -> None:
    set_seed(cfg["seed"])
    device = torch.device(cfg["training"]["device"])
    model_type = cfg["model"]["type"]

    transform = T.Compose([T.ToTensor(), T.Normalize([0.5] * 3, [0.5] * 3)])
    full_train = RafDbDataset(REPO_ROOT / cfg["data"]["dataset_root"], "train", transform=transform)
    test_ds = RafDbDataset(REPO_ROOT / cfg["data"]["dataset_root"], "test", transform=transform)

    labels = [s.label for s in full_train.samples]
    train_idx, val_idx = stratified_split(labels, cfg["data"]["val_fraction"], cfg["seed"])
    train_ds = Subset(full_train, train_idx)
    val_ds = Subset(full_train, val_idx)

    n_workers = cfg["training"]["num_workers"]
    train_loader = DataLoader(train_ds, batch_size=cfg["training"]["batch_size"], shuffle=True, num_workers=n_workers)
    val_loader = DataLoader(val_ds, batch_size=cfg["training"]["batch_size"], shuffle=False, num_workers=n_workers)
    test_loader = DataLoader(test_ds, batch_size=cfg["training"]["batch_size"], shuffle=False, num_workers=n_workers)

    class_weights = None
    if cfg["training"]["use_class_weights"]:
        counts = np.bincount(np.array(labels)[train_idx], minlength=N_CLASSES).astype(np.float32)
        weights = counts.sum() / (N_CLASSES * counts)
        class_weights = torch.tensor(weights, dtype=torch.float32)
        print("Class weights (inverse frequency):", weights)

    model = build_model(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["training"]["learning_rate"], weight_decay=cfg["training"]["weight_decay"])
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    results_dir = REPO_ROOT / cfg["output"]["results_dir"] / model_type
    embeddings_dir = REPO_ROOT / cfg["output"]["embeddings_dir"] / model_type
    checkpoints_dir = REPO_ROOT / cfg["output"]["checkpoints_dir"] / model_type
    for d in (results_dir, embeddings_dir, checkpoints_dir):
        d.mkdir(parents=True, exist_ok=True)

    best_val_bal_acc, best_state = -1.0, None
    t_start = time.time()
    for epoch in range(cfg["training"]["epochs"]):
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits, _ = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)

        model.eval()
        val_true, val_pred = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                logits, _ = model(xb)
                val_pred.append(logits.argmax(dim=-1).cpu().numpy())
                val_true.append(yb.numpy())
        val_metrics = classification_report_dict(np.concatenate(val_true), np.concatenate(val_pred), N_CLASSES)
        print(
            f"[{model_type}] epoch {epoch+1}/{cfg['training']['epochs']} "
            f"train_loss={epoch_loss/len(train_ds):.4f} val_acc={val_metrics['accuracy']:.4f} "
            f"val_balanced_acc={val_metrics['balanced_accuracy']:.4f}"
        )
        if val_metrics["balanced_accuracy"] > best_val_bal_acc:
            best_val_bal_acc = val_metrics["balanced_accuracy"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), checkpoints_dir / "model.pt")

    model.eval()
    test_true, test_pred, test_embed = [], [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            logits, embedding = model(xb)
            test_pred.append(logits.argmax(dim=-1).cpu().numpy())
            test_true.append(yb.numpy())
            test_embed.append(embedding.cpu().numpy())
    test_true, test_pred, test_embed = np.concatenate(test_true), np.concatenate(test_pred), np.concatenate(test_embed)

    test_metrics = classification_report_dict(test_true, test_pred, N_CLASSES)
    elapsed = time.time() - t_start
    test_metrics["elapsed_seconds"] = elapsed
    test_metrics["best_val_balanced_accuracy"] = best_val_bal_acc
    test_metrics["model_type"] = model_type

    np.savez(embeddings_dir / "test_embeddings.npz", embeddings=test_embed, labels=test_true, predictions=test_pred)
    with open(results_dir / "test_summary.json", "w") as f:
        json.dump(test_metrics, f, indent=2)

    print(f"\n=== {model_type} RAF-DB test results ({elapsed:.1f}s) ===")
    for k in ("accuracy", "balanced_accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_weighted"):
        print(f"  {k}: {test_metrics[k]:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "fer_raf_db.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    train_and_evaluate(cfg)
