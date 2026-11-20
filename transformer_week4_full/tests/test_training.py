# Author: [SunMuzihang] | Student ID: [2260031490]

"""Tests for the Week 3 data pipeline, loss, and learning-rate schedule."""

from __future__ import annotations

import math

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from data import ToySeq2SeqDataset, build_toy_vocab
from model import Transformer
from training import (
    LabelSmoothingCrossEntropy,
    NoamScheduler,
    load_checkpoint,
    save_checkpoint,
)


def build_tiny_model(vocab_size: int = 8) -> Transformer:
    """Build a small CPU model for training-pipeline tests."""
    return Transformer(
        vocab_size,
        vocab_size,
        d_model=16,
        num_heads=2,
        num_encoder_layers=1,
        num_decoder_layers=1,
        d_ff=32,
        dropout=0.0,
        max_len=8,
        share_embeddings=True,
        pad_idx=0,
    )


def test_vocab_special_tokens_and_roundtrip() -> None:
    vocab = build_toy_vocab(4)

    assert vocab.pad_idx == 0
    assert vocab.bos_idx == 1
    assert vocab.eos_idx == 2
    assert vocab.unk_idx == 3
    assert len(vocab) == 8

    ids = vocab.encode(["w0", "w3", "missing"], add_bos=True, add_eos=True)
    assert ids[0] == vocab.bos_idx
    assert ids[-1] == vocab.eos_idx
    assert ids[3] == vocab.unk_idx
    assert vocab.decode(ids) == ["w0", "w3", "<unk>"]


def test_toy_dataset_is_shifted_and_reversed() -> None:
    vocab = build_toy_vocab(8)
    dataset = ToySeq2SeqDataset(
        vocab, num_samples=8, min_len=3, max_len=5, reverse=True, seed=0
    )

    src, tgt_input, tgt_output = dataset[0]

    assert src.dtype == torch.long
    assert tgt_input[0].item() == vocab.bos_idx
    assert tgt_output[-1].item() == vocab.eos_idx
    # Decoder input and labels are shifted versions of the same full target.
    assert torch.equal(tgt_input[1:], tgt_output[:-1])
    # The target body is the reversed source.
    assert torch.equal(torch.flip(src, dims=[0]), tgt_output[:-1])


def test_collate_fn_pads_batches() -> None:
    vocab = build_toy_vocab(8)
    dataset = ToySeq2SeqDataset(
        vocab, num_samples=8, min_len=2, max_len=6, seed=0
    )
    loader = DataLoader(
        dataset, batch_size=4, shuffle=False, collate_fn=dataset.collate_fn
    )

    src, tgt_input, tgt_output = next(iter(loader))

    assert src.shape[0] == 4
    assert src.shape[1] == max(len(dataset[i][0]) for i in range(4))
    assert tgt_input.shape == tgt_output.shape
    assert src.dtype == torch.long
    assert src[:, 0].ne(vocab.pad_idx).all()


@pytest.mark.parametrize("smoothing", [0.0, 0.1, 0.5])
def test_label_smoothing_matches_pytorch(smoothing: float) -> None:
    torch.manual_seed(0)
    logits = torch.randn(6, 9)
    target = torch.tensor([1, 0, 5, 3, 0, 8], dtype=torch.long)

    custom = LabelSmoothingCrossEntropy(smoothing=smoothing, ignore_index=0)
    reference = nn.CrossEntropyLoss(label_smoothing=smoothing, ignore_index=0)

    assert torch.allclose(custom(logits, target), reference(logits, target), atol=1e-6)


def test_label_smoothing_all_ignored_is_differentiable() -> None:
    logits = torch.randn(3, 7, requires_grad=True)
    target = torch.zeros(3, dtype=torch.long)
    criterion = LabelSmoothingCrossEntropy(smoothing=0.1, ignore_index=0)

    loss = criterion(logits, target)
    loss.backward()

    assert loss.item() == 0.0
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum() == 0


def test_noam_schedule_matches_formula() -> None:
    parameter = nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.SGD([parameter], lr=1.0)
    scheduler = NoamScheduler(
        optimizer, d_model=512, warmup_steps=4000, factor=1.0
    )

    expected_step_1 = 512 ** -0.5 * min(1.0, 1 * 4000 ** -1.5)
    assert math.isclose(
        scheduler.get_last_lr()[0], expected_step_1, rel_tol=1e-12
    )
    assert math.isclose(
        optimizer.param_groups[0]["lr"], expected_step_1, rel_tol=1e-12
    )

    returned_lr = scheduler.step()
    expected_step_2 = 512 ** -0.5 * min(2 ** -0.5, 2 * 4000 ** -1.5)
    assert math.isclose(returned_lr, expected_step_2, rel_tol=1e-12)
    assert math.isclose(
        scheduler.get_last_lr()[0], expected_step_2, rel_tol=1e-12
    )


def test_noam_schedule_peaks_at_warmup() -> None:
    parameter = nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.SGD([parameter], lr=1.0)
    scheduler = NoamScheduler(optimizer, d_model=512, warmup_steps=100)

    peak = scheduler.lr_at(100)
    assert math.isclose(peak, 512 ** -0.5 * 100 ** -0.5, rel_tol=1e-12)
    assert scheduler.lr_at(99) < peak
    assert scheduler.lr_at(101) < peak


def test_training_step_integration() -> None:
    torch.manual_seed(0)
    vocab = build_toy_vocab(8)
    dataset = ToySeq2SeqDataset(
        vocab, num_samples=8, min_len=3, max_len=5, seed=0
    )
    loader = DataLoader(
        dataset, batch_size=4, shuffle=False, collate_fn=dataset.collate_fn
    )

    model = build_tiny_model(len(vocab))
    criterion = LabelSmoothingCrossEntropy(
        smoothing=0.1, ignore_index=vocab.pad_idx
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9
    )
    scheduler = NoamScheduler(optimizer, d_model=16, warmup_steps=10)

    src, tgt_input, tgt_output = next(iter(loader))
    logits = model(src, tgt_input)
    loss = criterion(
        logits.reshape(-1, logits.size(-1)),
        tgt_output.reshape(-1),
    )
    loss.backward()
    optimizer.step()
    scheduler.step()

    assert torch.isfinite(loss).item()
    assert loss.item() > 0.0
    for parameter in model.parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all()


def test_checkpoint_roundtrip(tmp_path) -> None:
    torch.manual_seed(0)
    model = build_tiny_model()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9
    )
    scheduler = NoamScheduler(optimizer, d_model=16, warmup_steps=10)
    scheduler.step()
    scheduler.step()

    path = save_checkpoint(
        tmp_path / "checkpoint.pt",
        model,
        optimizer,
        scheduler,
        step=scheduler.step_num,
    )
    assert path.exists()

    restored_model = build_tiny_model()
    restored_optimizer = torch.optim.Adam(
        restored_model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9
    )
    restored_scheduler = NoamScheduler(
        restored_optimizer, d_model=16, warmup_steps=10
    )
    checkpoint = load_checkpoint(
        path,
        restored_model,
        restored_optimizer,
        restored_scheduler,
    )

    assert checkpoint["step"] == scheduler.step_num
    assert restored_scheduler.step_num == scheduler.step_num
    assert math.isclose(
        restored_optimizer.param_groups[0]["lr"],
        scheduler.get_last_lr()[0],
        rel_tol=1e-12,
    )
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, restored_model.state_dict()[name])
