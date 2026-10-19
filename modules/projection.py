"""Residual feature projection."""

from __future__ import annotations

import torch.nn as nn


class ResidualProjection(nn.Module):
    def __init__(self, input_dim, output_dim, dropout):
        super().__init__()
        self.skip = nn.Linear(input_dim, output_dim)
        self.block = nn.Sequential(
            nn.Linear(input_dim, output_dim * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(output_dim * 2, output_dim), nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x):
        return self.norm(self.skip(x) + self.block(x))
