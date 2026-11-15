# Author: [SunMuzihang] | Student ID: [2260031490]

"""Toy sequence-to-sequence dataset used by the Week 3 training demo."""

from __future__ import annotations

import random
from typing import Sequence

import torch
from torch import Tensor
from torch.utils.data import Dataset

from .vocab import Vocab


class ToySeq2SeqDataset(Dataset):
    """A deterministic copy/reverse task for CPU training smoke tests.

    Each example contains:

    * ``src``: a random sequence of non-special token ids;
    * ``tgt_input``: ``[<bos>] + target_body`` (decoder input);
    * ``tgt_output``: ``target_body + [<eos>]`` (labels).

    By default the target body is the reversed source, so the model must learn
    a simple sequence transformation. Decoder inputs are already shifted right
    and labels are shifted left, matching the model's teacher-forcing contract.

    Args:
        vocab: Vocabulary used to build the examples.
        num_samples: Number of examples in the dataset.
        min_len: Minimum source length.
        max_len: Maximum source length.
        reverse: If true, the target is the reversed source; otherwise it is
            a copy of the source.
        seed: Seed for deterministic example generation.
    """

    def __init__(
        self,
        vocab: Vocab,
        num_samples: int = 128,
        min_len: int = 3,
        max_len: int = 8,
        reverse: bool = True,
        seed: int = 42,
    ) -> None:
        if num_samples < 1:
            raise ValueError(f"num_samples must be positive; got {num_samples}.")
        if min_len < 1:
            raise ValueError(f"min_len must be positive; got {min_len}.")
        if max_len < min_len:
            raise ValueError(
                f"max_len ({max_len}) must be greater than or equal to "
                f"min_len ({min_len})."
            )

        self.vocab = vocab
        self.num_samples = num_samples
        self.min_len = min_len
        self.max_len = max_len
        self.reverse = reverse
        self.pad_idx = vocab.pad_idx
        self.bos_idx = vocab.bos_idx
        self.eos_idx = vocab.eos_idx

        self._non_special_ids = [
            idx
            for idx, token in enumerate(vocab.itos)
            if token not in vocab.special_tokens
        ]
        if not self._non_special_ids:
            raise ValueError("The vocabulary must contain at least one non-special token.")

        self.samples = self._build_samples(seed)

    def _build_samples(
        self,
        seed: int,
    ) -> list[tuple[list[int], list[int], list[int]]]:
        rng = random.Random(seed)
        samples: list[tuple[list[int], list[int], list[int]]] = []
        for _ in range(self.num_samples):
            length = rng.randint(self.min_len, self.max_len)
            src = [rng.choice(self._non_special_ids) for _ in range(length)]
            body = list(reversed(src)) if self.reverse else list(src)
            full_target = [self.bos_idx] + body + [self.eos_idx]
            samples.append((src, full_target[:-1], full_target[1:]))
        return samples

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor]:
        src, tgt_input, tgt_output = self.samples[index]
        return (
            torch.tensor(src, dtype=torch.long),
            torch.tensor(tgt_input, dtype=torch.long),
            torch.tensor(tgt_output, dtype=torch.long),
        )

    def collate_fn(
        self,
        batch: Sequence[tuple[Tensor, Tensor, Tensor]],
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Pad a batch of variable-length examples with ``pad_idx``."""
        src, tgt_input, tgt_output = zip(*batch)
        return (
            torch.nn.utils.rnn.pad_sequence(
                src, batch_first=True, padding_value=self.pad_idx
            ),
            torch.nn.utils.rnn.pad_sequence(
                tgt_input, batch_first=True, padding_value=self.pad_idx
            ),
            torch.nn.utils.rnn.pad_sequence(
                tgt_output, batch_first=True, padding_value=self.pad_idx
            ),
        )

    def __repr__(self) -> str:
        return (
            f"ToySeq2SeqDataset(size={self.num_samples}, "
            f"min_len={self.min_len}, max_len={self.max_len}, "
            f"reverse={self.reverse})"
        )
