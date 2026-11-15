# Author: [SunMuzihang] | Student ID: [2260031490]

"""Label-smoothed cross-entropy for sequence-to-sequence training."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class LabelSmoothingCrossEntropy(nn.Module):
    """Cross-entropy with label smoothing (Szegedy et al., 2016).

    The smoothed target distribution is

    ``(1 - smoothing) * one_hot(target) + smoothing / num_classes``

    for every class, matching ``torch.nn.CrossEntropyLoss(label_smoothing=...)``.
    Positions whose target equals ``ignore_index`` are excluded from the loss.

    Args:
        smoothing: Label-smoothing value in ``[0, 1)``.
        ignore_index: Target value that should not contribute to the loss.
        reduction: ``"mean"``, ``"sum"``, or ``"none"``. With ``"none"``, the
            returned tensor contains one loss per non-ignored position.

    Shape:
        - ``logits``: ``(N, C)`` unnormalized scores.
        - ``target``: ``(N,)`` long tensor.
        - Output: scalar for ``"mean"``/``"sum"``, or ``(num_valid,)`` for
          ``"none"``.
    """

    def __init__(
        self,
        smoothing: float = 0.1,
        ignore_index: int = -100,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if not 0.0 <= smoothing < 1.0:
            raise ValueError(f"smoothing must be in [0, 1); got {smoothing}.")
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError(
                "reduction must be 'none', 'mean', or 'sum'; "
                f"got {reduction!r}."
            )
        self.smoothing = smoothing
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        """Compute the label-smoothed cross-entropy loss."""
        if logits.dim() != 2:
            raise ValueError(
                f"logits must be 2-D (N, C); got shape {tuple(logits.shape)}."
            )
        if target.dim() != 1:
            raise ValueError(
                f"target must be 1-D (N,); got shape {tuple(target.shape)}."
            )
        if logits.size(0) != target.size(0):
            raise ValueError(
                "logits and target must have the same batch size; got "
                f"{logits.size(0)} and {target.size(0)}."
            )
        if target.dtype != torch.long:
            raise ValueError(f"target must be torch.long; got {target.dtype}.")

        log_probs = F.log_softmax(logits, dim=-1)
        valid = target.ne(self.ignore_index)
        if not valid.any():
            # Keep the graph connected so backward() remains safe.
            return logits.sum() * 0.0

        valid_log_probs = log_probs[valid]
        valid_target = target[valid]
        nll = -valid_log_probs.gather(1, valid_target.unsqueeze(1)).squeeze(1)

        if self.smoothing == 0.0:
            loss = nll
        else:
            smooth_loss = -valid_log_probs.mean(dim=-1)
            loss = (1.0 - self.smoothing) * nll + self.smoothing * smooth_loss

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
