"""Model A (baseline) for static-image FER datasets (RAF-DB, FER2013, AffectNet)."""
from __future__ import annotations

import torch
import torch.nn as nn


class FERBaselineCNN(nn.Module):
    """Input: (batch, 3, H, W) RGB image. Output: (logits, embedding)."""

    def __init__(self, n_classes: int, embedding_dim: int = 128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.embed = nn.Linear(128, embedding_dim)
        self.classifier = nn.Linear(embedding_dim, n_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.features(x).flatten(1)
        embedding = torch.relu(self.embed(h))
        logits = self.classifier(embedding)
        return logits, embedding
