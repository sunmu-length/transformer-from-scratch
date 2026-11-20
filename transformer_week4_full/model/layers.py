# Author: [SunMuzihang] | Student ID: [2260031490]

"""Shared building blocks: Post-LN sublayers, cloning, and padding masks."""

from __future__ import annotations

import copy
from typing import Callable, Optional

from torch import Tensor, nn

Sublayer = Callable[[Tensor], Tensor]


def clones(module: nn.Module, n: int) -> nn.ModuleList:
    """Return ``n`` independent deep copies of ``module``.

    Deep copies are required because each Transformer layer must own a
    separate set of parameters.
    """
    if n < 1:
        raise ValueError(f"n must be at least 1; got {n}.")
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


def padding_mask(tokens: Tensor, pad_idx: Optional[int]) -> Optional[Tensor]:
    """Build a key-padding mask from token ids.

    Args:
        tokens: Token ids of shape ``(B, L)``.
        pad_idx: Padding token id, or ``None`` if the input has no padding.

    Returns:
        ``None`` when ``pad_idx`` is ``None``; otherwise a boolean tensor of
        shape ``(B, L)`` where ``True`` marks padding positions. This matches
        the mask convention used by :class:`~model.attention.MultiHeadAttention`.
    """
    if pad_idx is None:
        return None
    return tokens.eq(pad_idx)


class SublayerConnection(nn.Module):
    """Post-LN residual connection around a sublayer.

    Implements ``LayerNorm(x + dropout(sublayer(x)))`` exactly as in the
    original Transformer; ``eps=1e-6`` follows the paper's LayerNorm setting.

    Args:
        d_model: Model hidden dimension.
        dropout: Dropout probability applied to the sublayer output.
        eps: LayerNorm epsilon.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, eps: float = 1e-6) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model, eps=eps)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, sublayer: Sublayer) -> Tensor:
        """Apply ``sublayer`` to ``x`` and add the residual connection."""
        return self.norm(x + self.dropout(sublayer(x)))
