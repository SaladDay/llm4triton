"""
Feed-forward MLP workload used to exercise the evaluation harness.
"""

from __future__ import annotations

import torch
from torch import nn

BATCH_SIZE = 16
IN_FEATURES = 1024
HIDDEN_FEATURES = 4096
OUT_FEATURES = 1024


class FeedForward(nn.Module):
    """
    Two-layer MLP with GELU activation, representative of attention MLP blocks.
    """

    def __init__(
        self,
        in_features: int = IN_FEATURES,
        hidden_features: int = HIDDEN_FEATURES,
        out_features: int = OUT_FEATURES,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.lin2 = nn.Linear(hidden_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))
