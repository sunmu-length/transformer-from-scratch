# Author: [SunMuzihang] | Student ID: [2260031490]

"""Vocabulary utilities for the toy sequence-to-sequence task."""

from __future__ import annotations

from typing import Iterable, Sequence

PAD_TOKEN = "<pad>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"
SPECIAL_TOKENS = (PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN)


class Vocab:
    """A minimal token-to-id vocabulary with fixed special tokens.

    Args:
        tokens: Additional token strings. Duplicates are ignored.
        special_tokens: Special tokens placed at the start of the vocabulary.
            Must contain ``<pad>``, ``<bos>``, ``<eos>``, and ``<unk>``.

    Attributes:
        itos: Id-to-token list.
        stoi: Token-to-id dictionary.
        pad_idx: Id of the padding token.
        bos_idx: Id of the beginning-of-sequence token.
        eos_idx: Id of the end-of-sequence token.
        unk_idx: Id of the unknown token.
    """

    def __init__(
        self,
        tokens: Iterable[str] = (),
        special_tokens: Sequence[str] = SPECIAL_TOKENS,
    ) -> None:
        self.special_tokens = tuple(special_tokens)
        if len(set(self.special_tokens)) != len(self.special_tokens):
            raise ValueError("special_tokens must be unique.")

        self.itos: list[str] = list(self.special_tokens)
        seen = set(self.itos)
        for token in tokens:
            if token not in seen:
                self.itos.append(token)
                seen.add(token)

        self.stoi = {token: idx for idx, token in enumerate(self.itos)}
        for token in SPECIAL_TOKENS:
            if token not in self.stoi:
                raise ValueError(
                    f"special_tokens must contain {token!r}; "
                    f"got {self.special_tokens!r}."
                )

        self.pad_idx = self.stoi[PAD_TOKEN]
        self.bos_idx = self.stoi[BOS_TOKEN]
        self.eos_idx = self.stoi[EOS_TOKEN]
        self.unk_idx = self.stoi[UNK_TOKEN]

    def __len__(self) -> int:
        return len(self.itos)

    def __contains__(self, token: object) -> bool:
        return token in self.stoi

    def encode(
        self,
        tokens: Iterable[str],
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[int]:
        """Convert tokens to ids, mapping unknown tokens to ``<unk>``."""
        ids = [self.stoi.get(token, self.unk_idx) for token in tokens]
        if add_bos:
            ids.insert(0, self.bos_idx)
        if add_eos:
            ids.append(self.eos_idx)
        return ids

    def decode(
        self,
        ids: Iterable[int],
        skip_special_tokens: bool = True,
    ) -> list[str]:
        """Convert ids back to tokens.

        When ``skip_special_tokens`` is true, ``<pad>``, ``<bos>``, and
        ``<eos>`` are removed. ``<unk>`` is kept so unknown inputs remain
        visible in the decoded output.
        """
        ignored = {PAD_TOKEN, BOS_TOKEN, EOS_TOKEN}
        tokens = []
        for idx in ids:
            if not 0 <= idx < len(self.itos):
                raise IndexError(f"Token id {idx} is outside the vocabulary.")
            token = self.itos[idx]
            if skip_special_tokens and token in ignored:
                continue
            tokens.append(token)
        return tokens

    def token_to_id(self, token: str) -> int:
        """Return the id for ``token``, or ``unk_idx`` if it is unknown."""
        return self.stoi.get(token, self.unk_idx)

    def id_to_token(self, idx: int) -> str:
        """Return the token for ``idx``."""
        if not 0 <= idx < len(self.itos):
            raise IndexError(f"Token id {idx} is outside the vocabulary.")
        return self.itos[idx]

    def __repr__(self) -> str:
        return f"Vocab(size={len(self)}, special_tokens={self.special_tokens})"


def build_toy_vocab(num_tokens: int = 32) -> Vocab:
    """Build a vocabulary with ``num_tokens`` synthetic word tokens."""
    if num_tokens < 1:
        raise ValueError(f"num_tokens must be positive; got {num_tokens}.")
    return Vocab(tokens=(f"w{i}" for i in range(num_tokens)))
