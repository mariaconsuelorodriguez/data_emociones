"""Model C (proposed): per-window linear projection + Transformer encoder over the sequence.

Compared empirically against the CNN+BiLSTM baseline before being adopted as
the definitive candidate encoder for the future multimodal fusion (per the
plan's rule that a more complex architecture is never assumed to win by
default).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class _PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[: x.size(1)].unsqueeze(0)


class EEGTransformer(nn.Module):
    """Input: (batch, seq_len, n_bands, n_channels). Output: (logits, embedding)."""

    def __init__(
        self,
        n_bands: int,
        n_channels: int,
        n_classes: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        embedding_dim: int = 128,
    ):
        super().__init__()
        self.input_proj = nn.Linear(n_bands * n_channels, d_model)
        self.pos_encoding = _PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.embed = nn.Linear(d_model, embedding_dim)
        self.classifier = nn.Linear(embedding_dim, n_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, seq_len = x.shape[0], x.shape[1]
        flat = x.view(batch, seq_len, -1)
        projected = self.input_proj(flat)
        projected = self.pos_encoding(projected)
        encoded = self.transformer(projected)
        pooled = encoded.mean(dim=1)
        embedding = torch.relu(self.embed(pooled))
        logits = self.classifier(embedding)
        return logits, embedding
