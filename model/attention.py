"""Attention primitives for the Transformer (Vaswani et al., 2017).

All modules follow the ``batch_first`` convention, i.e. inputs and outputs
have shape ``(batch_size, sequence_length, d_model)``. Masks are boolean
tensors where ``True`` marks a position that must **not** be attended to.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor, nn


def causal_mask(size: int, device: Optional[torch.device] = None) -> Tensor:
    """Build a causal (look-ahead) mask for decoder self-attention.

    Args:
        size: Sequence length.
        device: Device on which to allocate the mask.

    Returns:
        A ``(size, size)`` boolean tensor where ``True`` marks future
        positions, i.e. the strict upper triangle.
    """
    return torch.triu(
        torch.ones(size, size, dtype=torch.bool, device=device), diagonal=1
    )


def scaled_dot_product_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    mask: Optional[Tensor] = None,
    dropout: Optional[nn.Dropout] = None,
) -> tuple[Tensor, Tensor]:
    """Scaled dot-product attention.

    ``Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V``

    Args:
        query: Query tensor of shape ``(..., tgt_len, d_k)``.
        key: Key tensor of shape ``(..., src_len, d_k)``.
        value: Value tensor of shape ``(..., src_len, d_v)``.
        mask: Optional boolean tensor broadcastable to
            ``(..., tgt_len, src_len)``. ``True`` marks positions that are
            masked out.
        dropout: Optional dropout applied to the attention weights.

    Returns:
        A tuple ``(output, weights)`` where ``output`` has shape
        ``(..., tgt_len, d_v)`` and ``weights`` has shape
        ``(..., tgt_len, src_len)``.
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask.to(torch.bool), float("-inf"))

    weights = torch.softmax(scores, dim=-1)
    # If a query row is fully masked, softmax produces NaNs. Zero those rows
    # so that padding does not leak into the rest of the network.
    weights = torch.nan_to_num(weights, nan=0.0)

    if dropout is not None:
        weights = dropout(weights)

    output = torch.matmul(weights, value)
    return output, weights


class MultiHeadAttention(nn.Module):
    """Multi-head attention as described in Section 3.2.2 of the paper.

    The same module supports both use cases required by the Transformer:

    * **Self-attention** — pass the same tensor as ``query``, ``key`` and
      ``value`` (encoder layers, and decoder layers with a causal mask).
    * **Cross-attention** — pass decoder states as ``query`` and encoder
      outputs as ``key`` and ``value``.

    Args:
        d_model: Model hidden dimension.
        num_heads: Number of attention heads. ``d_model`` must be divisible
            by ``num_heads``.
        dropout: Dropout probability applied to the attention weights.
        bias: Whether the linear projections use a bias term.

    Shape:
        - Input: ``query`` ``(B, L, d_model)``, ``key``/``value``
          ``(B, S, d_model)``.
        - Output: ``(B, L, d_model)``.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by num_heads ({num_heads})."
            )

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

        self.attn_dropout = nn.Dropout(dropout)

    def _split_heads(self, x: Tensor) -> Tensor:
        """Reshape ``(B, L, d_model)`` into ``(B, H, L, d_k)``."""
        batch_size, seq_len, _ = x.size()
        return x.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        need_weights: bool = False,
    ) -> tuple[Tensor, Optional[Tensor]]:
        """Apply multi-head attention.

        Args:
            query: ``(B, L, d_model)``.
            key: ``(B, S, d_model)``.
            value: ``(B, S, d_model)``.
            key_padding_mask: ``(B, S)`` boolean mask, ``True`` for padding
                tokens that should be ignored as keys.
            attn_mask: ``(L, S)`` or ``(B, H, L, S)`` boolean mask, ``True``
                for positions that should be ignored. For decoder
                self-attention, pass ``causal_mask(L)``.
            need_weights: If ``True``, also return attention weights averaged
                over heads, with shape ``(B, L, S)``. Useful for
                visualization; disabled by default for efficiency.

        Returns:
            ``(output, weights)`` where ``weights`` is ``None`` unless
            ``need_weights`` is ``True``.
        """
        batch_size, tgt_len, _ = query.size()
        src_len = key.size(1)

        q = self._split_heads(self.q_proj(query))
        k = self._split_heads(self.k_proj(key))
        v = self._split_heads(self.v_proj(value))

        mask: Optional[Tensor] = None
        if attn_mask is not None:
            mask = attn_mask.to(torch.bool)
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, L, S)
            elif mask.dim() != 4:
                raise ValueError(
                    "attn_mask must have shape (L, S) or (B, H, L, S); "
                    f"got {tuple(mask.shape)}."
                )

        if key_padding_mask is not None:
            if key_padding_mask.dim() != 2:
                raise ValueError(
                    "key_padding_mask must have shape (B, S); "
                    f"got {tuple(key_padding_mask.shape)}."
                )
            padding = key_padding_mask.to(torch.bool).unsqueeze(1).unsqueeze(2)
            mask = padding if mask is None else mask | padding

        output, weights = scaled_dot_product_attention(
            q, k, v, mask=mask, dropout=self.attn_dropout
        )

        output = output.transpose(1, 2).contiguous().view(
            batch_size, tgt_len, self.d_model
        )
        output = self.out_proj(output)

        if need_weights:
            # Average over heads for a single interpretable attention map.
            weights = weights.mean(dim=1)
        else:
            weights = None

        return output, weights
