# Author: [SunMuzihang] | Student ID: [2260031490]

"""Forward-pass tests for the Week 2 Transformer implementation."""

from __future__ import annotations

import pytest
import torch

from model import MultiHeadAttention, Transformer, padding_mask

SRC_VOCAB = 50
TGT_VOCAB = 60
D_MODEL = 32


def build_model(**overrides) -> Transformer:
    """Build a small deterministic Transformer for forward tests."""
    kwargs = dict(
        src_vocab_size=SRC_VOCAB,
        tgt_vocab_size=TGT_VOCAB,
        d_model=D_MODEL,
        num_heads=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        d_ff=64,
        dropout=0.0,
        max_len=64,
        positional_encoding="sinusoidal",
        share_embeddings=False,
        pad_idx=0,
    )
    kwargs.update(overrides)
    return Transformer(**kwargs).eval()


@pytest.fixture(autouse=True)
def _seed() -> None:
    torch.manual_seed(0)


@pytest.mark.parametrize("positional_encoding", ["sinusoidal", "learned"])
def test_forward_shape_and_finiteness(positional_encoding: str) -> None:
    model = build_model(positional_encoding=positional_encoding)
    src = torch.randint(1, SRC_VOCAB, (2, 7))
    tgt = torch.randint(1, TGT_VOCAB, (2, 5))

    logits = model(src, tgt)

    assert logits.shape == (2, 5, TGT_VOCAB)
    assert logits.dtype == torch.float32
    assert torch.isfinite(logits).all()


def test_encode_decode_shapes() -> None:
    model = build_model()
    src = torch.randint(1, SRC_VOCAB, (2, 7))
    tgt = torch.randint(1, TGT_VOCAB, (2, 5))

    memory = model.encode(src)
    hidden = model.decode(
        tgt, memory, memory_key_padding_mask=padding_mask(src, 0)
    )

    assert memory.shape == (2, 7, D_MODEL)
    assert hidden.shape == (2, 5, D_MODEL)


def test_causal_mask_invariance() -> None:
    model = build_model(pad_idx=None)
    src = torch.randint(1, SRC_VOCAB, (2, 7))
    tgt = torch.randint(1, TGT_VOCAB, (2, 6))

    logits_a = model(src, tgt)
    tgt_changed = tgt.clone()
    tgt_changed[:, 3] = (tgt_changed[:, 3] + 1) % TGT_VOCAB
    logits_b = model(src, tgt_changed)

    assert torch.allclose(logits_a[:, :3], logits_b[:, :3], atol=1e-6)
    assert not torch.allclose(logits_a[:, 3:], logits_b[:, 3:])


def test_padding_does_not_affect_real_outputs() -> None:
    model = build_model()
    src = torch.randint(1, SRC_VOCAB, (2, 6))
    tgt = torch.randint(1, TGT_VOCAB, (2, 5))
    src_padded = torch.cat([src, torch.zeros(2, 2, dtype=torch.long)], dim=1)

    logits_real = model(src, tgt)
    logits_padded = model(src_padded, tgt)

    assert torch.allclose(logits_real, logits_padded, atol=1e-5, rtol=1e-4)


def test_fully_padded_rows_are_finite() -> None:
    model = build_model()
    src = torch.randint(1, SRC_VOCAB, (2, 7))
    tgt = torch.randint(1, TGT_VOCAB, (2, 5))
    src[0] = 0
    tgt[0] = 0

    logits = model(src, tgt)

    assert torch.isfinite(logits).all()


def test_gradients_flow() -> None:
    model = build_model()
    model.train()
    src = torch.randint(1, SRC_VOCAB, (2, 7))
    tgt = torch.randint(1, TGT_VOCAB, (2, 5))
    model.zero_grad(set_to_none=True)

    model(src, tgt).sum().backward()

    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
    assert model.encoder.layers[0].self_attn.q_proj.weight.grad.abs().sum() > 0
    assert model.tgt_embedding.token_embedding.weight.grad is not None


def test_learned_pe_participates_in_gradients() -> None:
    model = build_model(positional_encoding="learned")
    model.train()
    src = torch.randint(1, SRC_VOCAB, (2, 7))
    tgt = torch.randint(1, TGT_VOCAB, (2, 5))
    model.zero_grad(set_to_none=True)

    model(src, tgt).sum().backward()

    grad = model.src_embedding.positional_encoding.positions.weight.grad
    assert grad is not None
    assert torch.isfinite(grad).all()
    assert grad.abs().sum() > 0


def test_weight_tying() -> None:
    shared = build_model(
        src_vocab_size=SRC_VOCAB,
        tgt_vocab_size=SRC_VOCAB,
        share_embeddings=True,
    )
    assert shared.generator.weight is shared.tgt_embedding.token_embedding.weight
    assert (
        shared.src_embedding.token_embedding.weight
        is shared.tgt_embedding.token_embedding.weight
    )

    unshared = build_model()
    assert unshared.generator.weight is unshared.tgt_embedding.token_embedding.weight
    assert (
        unshared.src_embedding.token_embedding.weight
        is not unshared.tgt_embedding.token_embedding.weight
    )

    with pytest.raises(ValueError, match="share_embeddings"):
        build_model(share_embeddings=True)


def test_no_padding_idx() -> None:
    model = build_model(pad_idx=None)
    src = torch.randint(0, SRC_VOCAB, (2, 7))
    tgt = torch.randint(0, TGT_VOCAB, (2, 5))

    logits = model(src, tgt)

    assert logits.shape == (2, 5, TGT_VOCAB)
    assert torch.isfinite(logits).all()
    assert padding_mask(src, None) is None


def test_input_validation() -> None:
    model = build_model(max_len=8)
    src = torch.randint(1, SRC_VOCAB, (2, 7))
    tgt = torch.randint(1, TGT_VOCAB, (2, 5))

    with pytest.raises(ValueError, match="max_len"):
        model(torch.randint(1, SRC_VOCAB, (2, 9)), tgt)
    with pytest.raises(ValueError, match="2-D"):
        model(src.unsqueeze(-1), tgt)
    with pytest.raises(ValueError, match="batch"):
        model(src, torch.randint(1, TGT_VOCAB, (3, 5)))
    with pytest.raises(ValueError, match="torch.long"):
        model(src.float(), tgt)


def test_mha_need_weights() -> None:
    mha = MultiHeadAttention(D_MODEL, 4, dropout=0.0).eval()
    x = torch.randn(2, 5, D_MODEL)

    output, weights = mha(x, x, x)
    assert output.shape == (2, 5, D_MODEL)
    assert weights is None

    output, weights = mha(x, x, x, need_weights=True)
    assert output.shape == (2, 5, D_MODEL)
    assert weights is not None
    assert weights.shape == (2, 5, 5)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(2, 5), atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cuda_forward() -> None:
    model = build_model().to("cuda")
    src = torch.randint(1, SRC_VOCAB, (2, 7), device="cuda")
    tgt = torch.randint(1, TGT_VOCAB, (2, 5), device="cuda")

    logits = model(src, tgt)

    assert logits.shape == (2, 5, TGT_VOCAB)
    assert logits.device.type == "cuda"
    assert torch.isfinite(logits).all()
