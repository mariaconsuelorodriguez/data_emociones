"""Model B (advanced) for PPG: CNN1D encoder (temporally resolved) + BiLSTM."""
from __future__ import annotations

import torch
import torch.nn as nn


class PPGCNNBiLSTM(nn.Module):
    """Input: (batch, 1, window_len). Output: (logits, embedding)."""

    def __init__(self, n_classes: int, cnn_dim: int = 64, lstm_hidden: int = 64, embedding_dim: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=15, padding=7),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(4),
            nn.Conv1d(32, 48, kernel_size=9, padding=4),
            nn.BatchNorm1d(48),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(4),
            nn.Conv1d(48, cnn_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(cnn_dim),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(4),
        )
        self.lstm = nn.LSTM(input_size=cnn_dim, hidden_size=lstm_hidden, num_layers=1, batch_first=True, bidirectional=True)
        self.embed = nn.Linear(2 * lstm_hidden, embedding_dim)
        self.classifier = nn.Linear(embedding_dim, n_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder(x).transpose(1, 2)  # (batch, reduced_time, cnn_dim)
        _, (h_n, _) = self.lstm(features)
        pooled = torch.cat([h_n[0], h_n[1]], dim=-1)
        embedding = torch.relu(self.embed(pooled))
        logits = self.classifier(embedding)
        return logits, embedding
