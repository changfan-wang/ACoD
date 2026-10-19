"""Gene-expression decoder."""

from __future__ import annotations

import torch.nn as nn


class FullGeneDecoder(nn.Module):
    def __init__(self, input_dim, gene_dim, hidden_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout), nn.Linear(hidden_dim, gene_dim),
        )

    def forward(self, x):
        return self.net(x)
