# Author: [SunMuzihang] | Student ID: [2260031490]

"""Token embeddings and positional encodings.

Two positional-encoding variants are provided so that the ablation in the
README can swap them without touching the rest of the model:

* :class:`PositionalEncoding` — the fixed sinusoidal encoding from the paper.
* :class:`LearnedPositionalEncoding` — a trainable embedding per position.

Both follow the ``batch_first`` convention: ``(B, L, d_model)``.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor, nn


class PositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding (Vaswani et al., 2017, §3.5).

    ``PE[pos, 2i]   = sin(pos / 10000^(2i / d_model))``
    ``PE[pos, 2i+1] = cos(pos / 10000^(2i / d_model))``

    The encoding is stored as a non-persistent buffer, so it moves with the
    module across devices but is not duplicated in checkpoints.

    Args:
        d_model: Model hidden dimension.
        dropout: Dropout applied after adding the encoding.
        max_len: Maximum supported sequence length.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        # Slicing keeps this correct for odd d_model as well.
        pe[:, 1::2] = torch.cos(position * div_term)[:, : pe[:, 1::2].shape[1]]

        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        """Add positional information to ``x`` of shape ``(B, L, d_model)``."""
        seq_len = x.size(1)
        if seq_len > self.pe.size(1):
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_len={self.pe.size(1)}."
            )
        x = x + self.pe[:, :seq_len]
        return self.dropout(x)


class LearnedPositionalEncoding(nn.Module):
    """Trainable positional embeddings, used as the ablation baseline.

    Args:
        d_model: Model hidden dimension.
        dropout: Dropout applied after adding the encoding.
        max_len: Maximum supported sequence length.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.max_len = max_len
        self.positions = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)

        nn.init.normal_(self.positions.weight, mean=0.0, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        """Add learned positional embeddings to ``x`` of shape ``(B, L, d_model)``."""
        seq_len = x.size(1)
        if seq_len > self.max_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_len={self.max_len}."
            )
        positions = torch.arange(seq_len, device=x.device)
        return self.dropout(x + self.positions(positions))


class TransformerEmbedding(nn.Module):
    """Token embedding scaled by ``sqrt(d_model)`` plus positional encoding.

    Scaling the embedding by ``sqrt(d_model)`` keeps its magnitude comparable
    to the positional signal, as in the original implementation.

    Args:
        vocab_size: Size of the token vocabulary.
        d_model: Model hidden dimension.
        max_len: Maximum supported sequence length.
        dropout: Dropout probability.
        positional_encoding: ``"sinusoidal"`` (paper default) or ``"learned"``.
        padding_idx: Vocabulary index of the padding token, if any. Its
            embedding is fixed to zero and receives no gradient.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        max_len: int = 5000,
        dropout: float = 0.1,
        positional_encoding: str = "sinusoidal",
        padding_idx: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(
            vocab_size, d_model, padding_idx=padding_idx
        )
        self.scale = math.sqrt(d_model)

        if positional_encoding == "sinusoidal":
            self.positional_encoding: nn.Module = PositionalEncoding(
                d_model, dropout, max_len
            )
        elif positional_encoding == "learned":
            self.positional_encoding = LearnedPositionalEncoding(
                d_model, dropout, max_len
            )
        else:
            raise ValueError(
                "positional_encoding must be 'sinusoidal' or 'learned'; "
                f"got {positional_encoding!r}."
            )

    def forward(self, tokens: Tensor) -> Tensor:
        """Embed ``tokens`` of shape ``(B, L)`` into ``(B, L, d_model)``."""
        x = self.token_embedding(tokens) * self.scale
        return self.positional_encoding(x)
