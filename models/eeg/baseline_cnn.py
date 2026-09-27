"""Model A (baseline): a small 2D CNN over a single (bands x channels) DE window."""
from __future__ import annotations

import torch
import torch.nn as nn


class EEGBaselineCNN(nn.Module):
    """Input: (batch, 1, n_bands, n_channels). Output: (logits, embedding)."""

    def __init__(self, n_bands: int, n_channels: int, n_classes: int, embedding_dim: int = 128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3, 5), padding=(1, 2)),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=(3, 5), padding=(1, 2)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.embed = nn.Linear(32, embedding_dim)
        self.classifier = nn.Linear(embedding_dim, n_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.dim() == 3:
            x = x.unsqueeze(1)  # (B, 1, bands, channels)
        h = self.features(x).flatten(1)
        embedding = torch.relu(self.embed(h))
        logits = self.classifier(embedding)
        return logits, embedding
