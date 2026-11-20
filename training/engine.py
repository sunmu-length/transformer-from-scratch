# Author: [SunMuzihang] | Student ID: [2260031490]

"""Small training/evaluation engine shared by scripts and experiments."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from .metrics import cross_entropy_loss, sequence_exact_match, token_accuracy
from .scheduler import NoamScheduler


def train_steps(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Optimizer,
    scheduler: NoamScheduler,
    criterion: nn.Module,
    steps: int,
    device: str | torch.device = "cpu",
) -> list[float]:
    """Run ``steps`` optimizer updates, iterating over the loader as needed.

    Unlike ``scripts/train.py``, this helper can train for more steps than a
    single epoch contains, which the ablation runner needs.
    """
    if steps < 1:
        raise ValueError(f"steps must be positive; got {steps}.")
    if len(loader) == 0:
        raise ValueError("train_steps() received an empty loader.")

    device = torch.device(device)
    model.train()
    losses: list[float] = []

    while len(losses) < steps:
        for src, tgt_input, tgt_output in loader:
            src = src.to(device)
            tgt_input = tgt_input.to(device)
            tgt_output = tgt_output.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(src, tgt_input)
            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                tgt_output.reshape(-1),
            )
            loss.backward()
            optimizer.step()
            scheduler.step()

            losses.append(float(loss.item()))
            if len(losses) >= steps:
                break

    return losses


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    pad_idx: int,
    device: str | torch.device = "cpu",
) -> dict[str, float]:
    """Evaluate loss, token accuracy, and exact match on ``loader``."""
    if len(loader) == 0:
        raise ValueError("evaluate() received an empty loader.")

    device = torch.device(device)
    model.eval()
    logits_batches = []
    target_batches = []

    for src, tgt_input, tgt_output in loader:
        src = src.to(device)
        tgt_input = tgt_input.to(device)
        logits = model(src, tgt_input)
        logits_batches.append(logits.cpu())
        target_batches.append(tgt_output.cpu())

    # Batches can have different padded target lengths, so pad them to a
    # common length before concatenating. The padded target positions use
    # ``pad_idx`` and are ignored by every metric.
    max_len = max(logits.size(1) for logits in logits_batches)
    logits = torch.cat(
        [
            F.pad(logits, (0, 0, 0, max_len - logits.size(1)))
            for logits in logits_batches
        ],
        dim=0,
    )
    target = torch.cat(
        [
            F.pad(target, (0, max_len - target.size(1)), value=pad_idx)
            for target in target_batches
        ],
        dim=0,
    )
    return {
        "loss": cross_entropy_loss(logits, target, ignore_index=pad_idx),
        "token_accuracy": token_accuracy(logits, target, ignore_index=pad_idx),
        "exact_match": sequence_exact_match(
            logits, target, ignore_index=pad_idx
        ),
    }
