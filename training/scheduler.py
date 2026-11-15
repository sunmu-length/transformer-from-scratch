# Author: [SunMuzihang] | Student ID: [2260031490]

"""The Noam learning-rate schedule from Vaswani et al. (2017)."""

from __future__ import annotations

from typing import Any

from torch.optim import Optimizer


class NoamScheduler:
    """Warmup + inverse-square-root learning-rate schedule.

    ``lr = factor * d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))``

    The optimizer's initial learning rate is overwritten immediately, so the
    scheduler controls the absolute learning rate. Call :meth:`step` after each
    ``optimizer.step()``; the first optimizer update uses step 1.

    Args:
        optimizer: Optimizer whose parameter groups will be updated.
        d_model: Model hidden dimension.
        warmup_steps: Number of warmup steps before the peak learning rate.
        factor: Optional scale factor; the paper uses ``1.0``.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        d_model: int,
        warmup_steps: int = 4000,
        factor: float = 1.0,
    ) -> None:
        if d_model < 1:
            raise ValueError(f"d_model must be positive; got {d_model}.")
        if warmup_steps < 1:
            raise ValueError(f"warmup_steps must be positive; got {warmup_steps}.")
        if factor <= 0.0:
            raise ValueError(f"factor must be positive; got {factor}.")

        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.factor = factor
        self.step_num = 1
        self._set_lr(self.lr_at(self.step_num))

    def lr_at(self, step: int) -> float:
        """Return the learning rate for a 1-indexed ``step``."""
        step = max(1, int(step))
        return (
            self.factor
            * self.d_model ** -0.5
            * min(step ** -0.5, step * self.warmup_steps ** -1.5)
        )

    def _set_lr(self, lr: float) -> None:
        for group in self.optimizer.param_groups:
            group["lr"] = lr

    def step(self) -> float:
        """Advance to the next step and return its learning rate."""
        self.step_num += 1
        lr = self.lr_at(self.step_num)
        self._set_lr(lr)
        return lr

    def get_last_lr(self) -> list[float]:
        """Return the learning rate currently stored in each parameter group."""
        return [float(group["lr"]) for group in self.optimizer.param_groups]

    def state_dict(self) -> dict[str, Any]:
        """Return scheduler state for checkpointing."""
        return {
            "step_num": self.step_num,
            "d_model": self.d_model,
            "warmup_steps": self.warmup_steps,
            "factor": self.factor,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore scheduler state from :meth:`state_dict`."""
        self.step_num = int(state_dict["step_num"])
        self._set_lr(self.lr_at(self.step_num))

    def __repr__(self) -> str:
        return (
            f"NoamScheduler(d_model={self.d_model}, "
            f"warmup_steps={self.warmup_steps}, factor={self.factor}, "
            f"step_num={self.step_num})"
        )
