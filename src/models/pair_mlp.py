from __future__ import annotations

import torch
import torch.nn as nn


class PairMLP(nn.Module):
    def __init__(self, in_dim: int, n_types: int, hidden: int = 512,
                 layers: int = 2, dropout: float = 0.2):
        super().__init__()
        trunk = []
        d = in_dim
        for _ in range(layers):
            trunk += [nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(dropout)]
            d = hidden
        self.trunk = nn.Sequential(*trunk)
        self.head_contact = nn.Linear(d, 1)
        self.head_types = nn.Linear(d, n_types)
        self.head_dist = nn.Linear(d, 1)

    def forward(self, x):
        h = self.trunk(x)
        return (
            self.head_contact(h).squeeze(-1),
            self.head_types(h),
            self.head_dist(h).squeeze(-1),
        )
