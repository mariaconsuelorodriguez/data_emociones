"""Model A (baseline) for EMG: MLP on the hand-crafted features from preprocessing/emg/features.py."""
from __future__ import annotations

import torch
import torch.nn as nn


class EMGBaselineMLP(nn.Module):
    """Input: (batch, n_features) e.g. the 9 features in preprocessing/emg/features.py,
    per channel and concatenated across channels. Output: (logits, embedding)."""

    def __init__(self, n_features: int, n_classes: int, embedding_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, embedding_dim),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(embedding_dim, n_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = self.net(x)
        logits = self.classifier(embedding)
        return logits, embedding
