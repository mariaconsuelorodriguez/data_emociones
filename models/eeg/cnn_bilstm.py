"""Model B (advanced): per-window CNN feature extractor + BiLSTM over the sequence."""
from __future__ import annotations

import torch
import torch.nn as nn


class _WindowEncoder(nn.Module):
    """Shared CNN applied independently to every window in a sequence."""

    def __init__(self, out_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3, 5), padding=(1, 2)),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, out_dim, kernel_size=(3, 5), padding=(1, 2)),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).flatten(1)


class EEGCNNBiLSTM(nn.Module):
    """Input: (batch, seq_len, n_bands, n_channels). Output: (logits, embedding)."""

    def __init__(
        self,
        n_bands: int,
        n_channels: int,
        n_classes: int,
        cnn_dim: int = 64,
        lstm_hidden: int = 64,
        embedding_dim: int = 128,
    ):
        super().__init__()
        self.encoder = _WindowEncoder(out_dim=cnn_dim)
        self.lstm = nn.LSTM(
            input_size=cnn_dim, hidden_size=lstm_hidden, num_layers=1, batch_first=True, bidirectional=True
        )
        self.embed = nn.Linear(2 * lstm_hidden, embedding_dim)
        self.classifier = nn.Linear(embedding_dim, n_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, seq_len = x.shape[0], x.shape[1]
        x = x.view(batch * seq_len, 1, x.shape[2], x.shape[3])
        per_window = self.encoder(x).view(batch, seq_len, -1)
        _, (h_n, _) = self.lstm(per_window)
        pooled = torch.cat([h_n[0], h_n[1]], dim=-1)  # forward + backward last states
        embedding = torch.relu(self.embed(pooled))
        logits = self.classifier(embedding)
        return logits, embedding
