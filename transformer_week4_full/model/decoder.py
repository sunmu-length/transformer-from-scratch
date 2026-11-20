# Author: [SunMuzihang] | Student ID: [2260031490]

"""Decoder layer and stack (Vaswani et al., 2017, §3.1)."""

from __future__ import annotations

from typing import Optional

from torch import Tensor, nn

from .attention import MultiHeadAttention
from .feed_forward import PositionwiseFeedForward
from .layers import SublayerConnection, clones


class DecoderLayer(nn.Module):
    """One decoder layer: masked self-attention, cross-attention, and FFN.

    Each sublayer uses a Post-LN residual connection:
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
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.sublayer = clones(SublayerConnection(d_model, dropout, eps), 3)

    def forward(
        self,
        x: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Decode ``x`` of shape ``(B, T, d_model)`` against ``memory``.

        Args:
            x: Target hidden states, ``(B, T, d_model)``.
            memory: Encoder outputs, ``(B, S, d_model)``.
            tgt_mask: Optional ``(T, T)`` causal mask; ``True`` = masked.
            tgt_key_padding_mask: Optional ``(B, T)`` target padding mask.
            memory_key_padding_mask: Optional ``(B, S)`` source padding mask.

        Returns:
            Decoder hidden states of shape ``(B, T, d_model)``.
        """
        # 1. Masked self-attention over the target prefix.
        x = self.sublayer[0](
            x,
            lambda h: self.self_attn(
                h,
                h,
                h,
                key_padding_mask=tgt_key_padding_mask,
                attn_mask=tgt_mask,
            )[0],
        )
        # 2. Cross-attention: queries from the decoder, keys/values from memory.
        x = self.sublayer[1](
            x,
            lambda h: self.cross_attn(
                h,
                memory,
                memory,
                key_padding_mask=memory_key_padding_mask,
            )[0],
        )
        # 3. Position-wise feed-forward network.
        return self.sublayer[2](x, self.feed_forward)


class Decoder(nn.Module):
    """A stack of ``num_layers`` identical decoder layers.

    Args:
        num_layers: Number of decoder layers.
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
        layer = DecoderLayer(d_model, num_heads, d_ff, dropout, eps)
        self.layers = clones(layer, num_layers)

    def forward(
        self,
        x: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Apply every decoder layer to ``x`` against ``memory``."""
        for layer in self.layers:
            x = layer(
                x,
                memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )
        return x
