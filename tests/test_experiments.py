# Author: [SunMuzihang] | Student ID: [2260031490]

"""Tests for Week 4 metrics, training engine, and attention extraction."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import ToySeq2SeqDataset, build_toy_vocab
from model import Transformer
from scripts.visualize_attention import extract_attention
from training import (
    LabelSmoothingCrossEntropy,
    NoamScheduler,
    cross_entropy_loss,
    evaluate,
    sequence_exact_match,
    token_accuracy,
    train_steps,
)


def build_tiny_model(
    vocab_size: int = 8,
    num_heads: int = 2,
) -> Transformer:
    return Transformer(
        vocab_size,
        vocab_size,
        d_model=16,
        num_heads=num_heads,
        num_encoder_layers=1,
        num_decoder_layers=1,
        d_ff=32,
        dropout=0.0,
        max_len=8,
        share_embeddings=True,
        pad_idx=0,
    )


def test_token_accuracy_and_exact_match() -> None:
    target = torch.tensor([[1, 2, 0], [3, 1, 0]])
    logits = torch.zeros(2, 3, 4)
    logits.scatter_(2, target.unsqueeze(-1), 10.0)

    assert token_accuracy(logits, target, ignore_index=0) == 1.0
    assert sequence_exact_match(logits, target, ignore_index=0) == 1.0

    # Make the first valid token of the first sequence wrong.
    logits[0, 0].zero_()
    logits[0, 0, 2] = 10.0

    assert math.isclose(
        token_accuracy(logits, target, ignore_index=0),
        0.75,
    )
    assert math.isclose(
        sequence_exact_match(logits, target, ignore_index=0),
        0.5,
    )

    # A wrong prediction at a padding position must not affect the metrics.
    logits[0, 2].zero_()
    logits[0, 2, 1] = 10.0
    assert math.isclose(
        token_accuracy(logits, target, ignore_index=0),
        0.75,
    )
    assert math.isclose(
        sequence_exact_match(logits, target, ignore_index=0),
        0.5,
    )


def test_cross_entropy_loss_matches_pytorch() -> None:
    torch.manual_seed(0)
    logits = torch.randn(2, 3, 4)
    target = torch.tensor([[1, 2, 0], [3, 1, 0]])

    got = cross_entropy_loss(logits, target, ignore_index=0)
    expected = F.cross_entropy(
        logits.reshape(-1, 4),
        target.reshape(-1),
        ignore_index=0,
    ).item()

    assert math.isclose(got, expected, rel_tol=1e-6)


def test_metric_validation() -> None:
    target = torch.tensor([[1, 2], [3, 1]])
    with pytest.raises(ValueError):
        token_accuracy(torch.randn(2, 2), target, ignore_index=0)
    with pytest.raises(ValueError):
        token_accuracy(torch.randn(2, 2, 4), target.float(), ignore_index=0)


def test_train_steps_runs_across_epochs_and_evaluate() -> None:
    torch.manual_seed(0)
    vocab = build_toy_vocab(8)
    dataset = ToySeq2SeqDataset(
        vocab, num_samples=4, min_len=2, max_len=4, seed=0
    )
    loader = DataLoader(
        dataset, batch_size=2, shuffle=False, collate_fn=dataset.collate_fn
    )

    model = build_tiny_model(len(vocab))
    criterion = LabelSmoothingCrossEntropy(
        smoothing=0.1, ignore_index=vocab.pad_idx
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9
    )
    scheduler = NoamScheduler(optimizer, d_model=16, warmup_steps=10)

    losses = train_steps(
        model,
        loader,
        optimizer,
        scheduler,
        criterion,
        steps=5,
        device="cpu",
    )

    assert len(losses) == 5
    assert all(math.isfinite(loss) for loss in losses)

    metrics = evaluate(model, loader, pad_idx=vocab.pad_idx, device="cpu")
    assert set(metrics) == {"loss", "token_accuracy", "exact_match"}
    assert math.isfinite(metrics["loss"])
    assert 0.0 <= metrics["token_accuracy"] <= 1.0
    assert 0.0 <= metrics["exact_match"] <= 1.0


def test_extract_attention_shapes() -> None:
    torch.manual_seed(0)
    model = build_tiny_model(num_heads=2).eval()
    src = torch.randint(1, 8, (1, 4))
    tgt_input = torch.randint(1, 8, (1, 3))

    self_weights, cross_weights = extract_attention(
        model,
        src,
        tgt_input,
        pad_idx=0,
        layer_index=-1,
    )

    assert self_weights.shape == (1, 3, 3)
    assert cross_weights.shape == (1, 3, 4)
    assert torch.allclose(
        self_weights.sum(dim=-1), torch.ones(1, 3), atol=1e-5
    )
    assert torch.allclose(
        cross_weights.sum(dim=-1), torch.ones(1, 3), atol=1e-5
    )
    assert self_weights.triu(diagonal=1).abs().max().item() == 0.0
