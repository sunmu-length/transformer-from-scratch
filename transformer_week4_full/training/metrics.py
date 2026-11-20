# Author: [SunMuzihang] | Student ID: [2260031490]

"""Evaluation metrics for the toy sequence-to-sequence task."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def _validate(
    logits: Tensor,
    target: Tensor,
    ignore_index: int,
) -> Tensor:
    if logits.dim() != 3:
        raise ValueError(
            f"logits must be 3-D (B, T, C); got shape {tuple(logits.shape)}."
        )
    if target.dim() != 2:
        raise ValueError(
            f"target must be 2-D (B, T); got shape {tuple(target.shape)}."
        )
    if logits.shape[:2] != target.shape:
        raise ValueError(
            "logits and target must share (B, T); got "
            f"{tuple(logits.shape[:2])} and {tuple(target.shape)}."
        )
    if target.dtype != torch.long:
        raise ValueError(f"target must be torch.long; got {target.dtype}.")
    return target.ne(ignore_index)


def token_accuracy(
    logits: Tensor,
    target: Tensor,
    ignore_index: int = -100,
) -> float:
    """Fraction of non-ignored target tokens predicted correctly."""
    mask = _validate(logits, target, ignore_index)
    total = int(mask.sum().item())
    if total == 0:
        return 0.0
    predictions = logits.argmax(dim=-1)
    correct = predictions.eq(target) & mask
    return float(correct.sum().item() / total)


def sequence_exact_match(
    logits: Tensor,
    target: Tensor,
    ignore_index: int = -100,
) -> float:
    """Fraction of sequences whose non-ignored tokens are all correct."""
    mask = _validate(logits, target, ignore_index)
    if target.size(0) == 0:
        return 0.0
    predictions = logits.argmax(dim=-1)
    matches = (predictions.eq(target) | ~mask).all(dim=1)
    return float(matches.float().mean().item())


def cross_entropy_loss(
    logits: Tensor,
    target: Tensor,
    ignore_index: int = -100,
) -> float:
    """Plain token-level cross-entropy, useful as a validation metric."""
    _validate(logits, target, ignore_index)
    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        target.reshape(-1),
        ignore_index=ignore_index,
    )
    return float(loss.item())
