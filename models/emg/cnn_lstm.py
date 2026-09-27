"""Model C (proposed) for EMG: CNN1D encoder (kept temporally resolved, not
fully pooled) + BiLSTM over the resulting time steps, per the plan's
"EMG-Transformer / CNN+LSTM, compared against simpler baselines" line.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class EMGCNNLSTM(nn.Module):
    """Input: (batch, n_channels, n_samples). Output: (logits, embedding)."""

    def __init__(self, n_channels: int, n_classes: int, cnn_dim: int = 64, lstm_hidden: int = 64, embedding_dim: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(n_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(4),
            nn.Conv1d(32, 48, kernel_size=5, padding=2),
            nn.BatchNorm1d(48),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(4),
            nn.Conv1d(48, cnn_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_dim),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(4),
        )
        self.lstm = nn.LSTM(input_size=cnn_dim, hidden_size=lstm_hidden, num_layers=1, batch_first=True, bidirectional=True)
        self.embed = nn.Linear(2 * lstm_hidden, embedding_dim)
        self.classifier = nn.Linear(embedding_dim, n_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder(x)  # (batch, cnn_dim, reduced_time)
        features = features.transpose(1, 2)  # (batch, reduced_time, cnn_dim)
        _, (h_n, _) = self.lstm(features)
        pooled = torch.cat([h_n[0], h_n[1]], dim=-1)
        embedding = torch.relu(self.embed(pooled))
        logits = self.classifier(embedding)
        return logits, embedding
