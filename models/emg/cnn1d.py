"""Model B (advanced) for EMG: 1D CNN directly on the segmented raw waveform."""
from __future__ import annotations

import torch
import torch.nn as nn


class EMGCNN1D(nn.Module):
    """Input: (batch, n_channels, n_samples) raw/segmented EMG. Output: (logits, embedding)."""

    def __init__(self, n_channels: int, n_classes: int, embedding_dim: int = 128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(n_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(4),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(4),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.embed = nn.Linear(128, embedding_dim)
        self.classifier = nn.Linear(embedding_dim, n_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.features(x).flatten(1)
        embedding = torch.relu(self.embed(h))
        logits = self.classifier(embedding)
        return logits, embedding
