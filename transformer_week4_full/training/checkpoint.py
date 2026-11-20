# Author: [SunMuzihang] | Student ID: [2260031490]

"""Minimal checkpoint helpers for the training loop."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import torch
from torch import nn
from torch.optim import Optimizer

from .scheduler import NoamScheduler


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: NoamScheduler,
    step: int,
    extra: Optional[dict[str, Any]] = None,
) -> Path:
    """Save model, optimizer, scheduler, and step metadata to ``path``."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    checkpoint: dict[str, Any] = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "step": int(step),
    }
    if extra is not None:
        checkpoint.update(extra)
    torch.save(checkpoint, destination)
    return destination


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[NoamScheduler] = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Restore a checkpoint created by :func:`save_checkpoint`.

    Only load checkpoint files from trusted sources: ``torch.load`` is called
    with ``weights_only=False`` so optimizer and scheduler state can be
    restored.
    """
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])
    return checkpoint
