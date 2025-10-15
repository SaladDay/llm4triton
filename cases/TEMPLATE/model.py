"""
Template PyTorch module for a new optimization case.

Copy this file alongside ``manifest.yaml`` and replace ``TemplateModel`` with the
original module you wish to optimize. No additional helper functions are
required—inputs and metadata are declared in ``manifest.yaml``.
"""

from __future__ import annotations

import torch
from torch import nn


class TemplateModel(nn.Module):
    """
    Replace this class body with your original PyTorch module.
    """

    def __init__(self) -> None:
        super().__init__()
        # Example parameter; remove once you paste your own implementation.
        self.linear = nn.Linear(16, 16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Replace with the real forward logic.
        return self.linear(x)
