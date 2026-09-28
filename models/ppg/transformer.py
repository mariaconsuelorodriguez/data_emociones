"""Model C (proposed) for PPG: patch embedding + Transformer encoder over the window.

The raw window (e.g. 1500 samples at the assumed 100 Hz) is split into
fixed-size patches (analogous to a vision transformer's patches), each
linearly projected to d_model, so self-attention operates over a short
sequence of patches instead of every individual sample.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class _PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 256):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[: x.size(1)].unsqueeze(0)


class PPGTransformer(nn.Module):
    """Input: (batch, 1, window_len). Output: (logits, embedding)."""

    def __init__(self, window_len: int, n_classes: int, patch_size: int = 50, d_model: int = 64, n_heads: int = 4, n_layers: int = 2, embedding_dim: int = 128):
        super().__init__()
        if window_len % patch_size != 0:
            raise ValueError(f"window_len ({window_len}) must be divisible by patch_size ({patch_size})")
        self.patch_size = patch_size
        self.n_patches = window_len // patch_size
        self.input_proj = nn.Linear(patch_size, d_model)
        self.pos_encoding = _PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.embed = nn.Linear(d_model, embedding_dim)
        self.classifier = nn.Linear(embedding_dim, n_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch = x.shape[0]
        patches = x.view(batch, self.n_patches, self.patch_size)
        projected = self.pos_encoding(self.input_proj(patches))
        encoded = self.transformer(projected)
        pooled = encoded.mean(dim=1)
        embedding = torch.relu(self.embed(pooled))
        logits = self.classifier(embedding)
        return logits, embedding
