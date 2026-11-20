# Author: [SunMuzihang] | Student ID: [2260031490]

"""Position-wise feed-forward network (Vaswani et al., 2017, §3.3)."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class PositionwiseFeedForward(nn.Module):
    """Two-layer MLP applied independently at every sequence position.

    ``FFN(x) = max(0, x W1 + b1) W2 + b2``

    Dropout is applied after the ReLU activation. The surrounding
    :class:`~model.layers.SublayerConnection` also applies dropout to the
    sublayer output before the residual addition, matching common reference
    implementations of the paper.

    Args:
        d_model: Model hidden dimension.
        d_ff: Hidden dimension of the feed-forward network.
        dropout: Dropout probability.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        """Apply the FFN to ``x`` of shape ``(B, L, d_model)``."""
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))
