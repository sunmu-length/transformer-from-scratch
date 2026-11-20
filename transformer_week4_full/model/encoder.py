# Author: [SunMuzihang] | Student ID: [2260031490]

"""Encoder layer and stack (Vaswani et al., 2017, §3.1)."""

from __future__ import annotations

from typing import Optional

from torch import Tensor, nn

from .attention import MultiHeadAttention
from .feed_forward import PositionwiseFeedForward
from .layers import SublayerConnection, clones


class EncoderLayer(nn.Module):
    """One encoder layer: self-attention followed by a feed-forward network.

    Both sublayers use a Post-LN residual connection:
    ``LayerNorm(x + dropout(sublayer(x)))``.

    Args:
        d_model: Model hidden dimension.
        num_heads: Number of attention heads.
        d_ff: Feed-forward hidden dimension.
        dropout: Dropout probability.
        eps: LayerNorm epsilon.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.sublayer = clones(SublayerConnection(d_model, dropout, eps), 2)

    def forward(
        self,
        x: Tensor,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Encode ``x`` of shape ``(B, S, d_model)``.

        ``src_key_padding_mask`` is ``(B, S)`` with ``True`` for padding keys.
        """
        x = self.sublayer[0](
            x,
            lambda h: self.self_attn(
                h,
                h,
                h,
                key_padding_mask=src_key_padding_mask,
            )[0],
        )
        return self.sublayer[1](x, self.feed_forward)


class Encoder(nn.Module):
    """A stack of ``num_layers`` identical encoder layers.

    Args:
        num_layers: Number of encoder layers.
        d_model: Model hidden dimension.
        num_heads: Number of attention heads.
        d_ff: Feed-forward hidden dimension.
        dropout: Dropout probability.
        eps: LayerNorm epsilon.
    """

    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        layer = EncoderLayer(d_model, num_heads, d_ff, dropout, eps)
        self.layers = clones(layer, num_layers)

    def forward(
        self,
        x: Tensor,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Apply every encoder layer to ``x`` of shape ``(B, S, d_model)``."""
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=src_key_padding_mask)
        return x
