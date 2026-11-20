# Author: [SunMuzihang] | Student ID: [2260031490]

"""End-to-end Transformer model (Vaswani et al., 2017)."""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn

from .attention import causal_mask
from .decoder import Decoder
from .embedding import TransformerEmbedding
from .encoder import Encoder
from .layers import padding_mask


class Transformer(nn.Module):
    """Full encoder–decoder Transformer.

    The model expects token ids and returns unnormalized logits. The target
    sequence must already be shifted right by the data pipeline; this module
    does not perform the shift.

    Args:
        src_vocab_size: Source vocabulary size.
        tgt_vocab_size: Target vocabulary size.
        d_model: Model hidden dimension.
        num_heads: Number of attention heads.
        num_encoder_layers: Number of encoder layers.
        num_decoder_layers: Number of decoder layers.
        d_ff: Feed-forward hidden dimension.
        dropout: Dropout probability.
        max_len: Maximum supported sequence length.
        positional_encoding: ``"sinusoidal"`` or ``"learned"``.
        share_embeddings: If ``True``, tie the source and target token
            embeddings. Requires equal vocabulary sizes. The output projection
            is always tied to the target embedding, as in the paper.
        pad_idx: Padding token id used to build padding masks automatically.

    Shape:
        - ``src_tokens``: ``(B, S)`` long tensor.
        - ``tgt_tokens``: ``(B, T)`` long tensor.
        - Output: ``(B, T, tgt_vocab_size)`` logits.
    """

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        d_ff: int = 2048,
        dropout: float = 0.1,
        max_len: int = 5000,
        positional_encoding: str = "sinusoidal",
        share_embeddings: bool = False,
        pad_idx: Optional[int] = None,
    ) -> None:
        super().__init__()
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive; got {num_heads}.")
        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by num_heads ({num_heads})."
            )
        if share_embeddings and src_vocab_size != tgt_vocab_size:
            raise ValueError(
                "share_embeddings=True requires src_vocab_size == tgt_vocab_size; "
                "use share_embeddings=False to share only the target embedding "
                "with the output projection."
            )
        if pad_idx is not None and not (
            0 <= pad_idx < min(src_vocab_size, tgt_vocab_size)
        ):
            raise ValueError(
                f"pad_idx={pad_idx} must be in "
                f"[0, {min(src_vocab_size, tgt_vocab_size)})."
            )

        self.d_model = d_model
        self.max_len = max_len
        self.pad_idx = pad_idx

        self.src_embedding = TransformerEmbedding(
            vocab_size=src_vocab_size,
            d_model=d_model,
            max_len=max_len,
            dropout=dropout,
            positional_encoding=positional_encoding,
            padding_idx=pad_idx,
        )
        self.tgt_embedding = TransformerEmbedding(
            vocab_size=tgt_vocab_size,
            d_model=d_model,
            max_len=max_len,
            dropout=dropout,
            positional_encoding=positional_encoding,
            padding_idx=pad_idx,
        )
        self.encoder = Encoder(
            num_layers=num_encoder_layers,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            dropout=dropout,
        )
        self.decoder = Decoder(
            num_layers=num_decoder_layers,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            dropout=dropout,
        )
        self.generator = nn.Linear(d_model, tgt_vocab_size, bias=False)

        self.apply(self._init_weights)

        # The paper shares the target embedding with the output projection.
        # Tie after initialization so each tensor is initialized exactly once.
        self.generator.weight = self.tgt_embedding.token_embedding.weight
        if share_embeddings:
            self.src_embedding.token_embedding.weight = (
                self.tgt_embedding.token_embedding.weight
            )

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        """Xavier-uniform Linear weights and normal-initialized embeddings."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].zero_()

    def _check_tokens(self, tokens: Tensor, name: str) -> None:
        """Validate a ``(B, L)`` token tensor before embedding lookup."""
        if tokens.dim() != 2:
            raise ValueError(
                f"{name} tokens must be 2-D (B, L); got shape {tuple(tokens.shape)}."
            )
        if tokens.dtype != torch.long:
            raise ValueError(
                f"{name} tokens must be torch.long; got {tokens.dtype}."
            )
        if tokens.size(1) > self.max_len:
            raise ValueError(
                f"{name} length {tokens.size(1)} exceeds max_len={self.max_len}."
            )

    def encode(
        self,
        src_tokens: Tensor,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Encode source tokens into memory of shape ``(B, S, d_model)``."""
        self._check_tokens(src_tokens, "src")
        if src_key_padding_mask is None:
            src_key_padding_mask = padding_mask(src_tokens, self.pad_idx)
        return self.encoder(
            self.src_embedding(src_tokens),
            src_key_padding_mask=src_key_padding_mask,
        )

    def decode(
        self,
        tgt_tokens: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Decode target tokens against encoder ``memory``.

        The causal mask and target padding mask are generated automatically
        when omitted. When calling ``encode`` and ``decode`` manually, pass
        ``memory_key_padding_mask`` explicitly if the source has padding.
        """
        self._check_tokens(tgt_tokens, "tgt")
        if tgt_mask is None:
            tgt_mask = causal_mask(tgt_tokens.size(1), device=tgt_tokens.device)
        if tgt_key_padding_mask is None:
            tgt_key_padding_mask = padding_mask(tgt_tokens, self.pad_idx)
        return self.decoder(
            self.tgt_embedding(tgt_tokens),
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )

    def forward(
        self,
        src_tokens: Tensor,
        tgt_tokens: Tensor,
        src_key_padding_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        tgt_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Run the full encoder–decoder forward pass.

        Returns logits of shape ``(B, T, tgt_vocab_size)``. The target tokens
        are expected to be shifted right already.
        """
        self._check_tokens(src_tokens, "src")
        self._check_tokens(tgt_tokens, "tgt")
        if src_tokens.size(0) != tgt_tokens.size(0):
            raise ValueError(
                "source and target batch sizes must match; got "
                f"{src_tokens.size(0)} and {tgt_tokens.size(0)}."
            )

        if src_key_padding_mask is None:
            src_key_padding_mask = padding_mask(src_tokens, self.pad_idx)
        if tgt_key_padding_mask is None:
            tgt_key_padding_mask = padding_mask(tgt_tokens, self.pad_idx)
        if tgt_mask is None:
            tgt_mask = causal_mask(tgt_tokens.size(1), device=tgt_tokens.device)
        if memory_key_padding_mask is None:
            memory_key_padding_mask = src_key_padding_mask

        memory = self.encode(
            src_tokens, src_key_padding_mask=src_key_padding_mask
        )
        hidden = self.decode(
            tgt_tokens,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        return self.generator(hidden)
