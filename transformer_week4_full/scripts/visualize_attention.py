# Author: [SunMuzihang] | Student ID: [2260031490]

"""Train a small model and plot decoder attention heatmaps.

Example:
    python scripts/visualize_attention.py --steps 300
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data import ToySeq2SeqDataset, build_toy_vocab
from model import Transformer, causal_mask, padding_mask
from training import LabelSmoothingCrossEntropy, NoamScheduler, train_steps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument(
        "--task",
        choices=("reverse", "copy"),
        default="reverse",
        help="Toy task: reverse the source or copy it.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-samples", type=int, default=2048)
    parser.add_argument("--min-len", type=int, default=3)
    parser.add_argument("--max-len", type=int, default=8)
    parser.add_argument("--num-tokens", type=int, default=32)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-encoder-layers", type=int, default=2)
    parser.add_argument("--num-decoder-layers", type=int, default=2)
    parser.add_argument("--d-ff", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument(
        "--layer",
        type=int,
        default=-1,
        help="Decoder layer to visualize; -1 means the last layer.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/attention_heatmap.png",
    )
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def extract_attention(
    model: Transformer,
    src: torch.Tensor,
    tgt_input: torch.Tensor,
    pad_idx: int,
    layer_index: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return last-layer self-attention and cross-attention weights.

    The decoder forward pass is replayed layer by layer so the attention
    weights are captured without changing the public ``Transformer.forward``
    API from Week 2.
    """
    device = src.device
    memory = model.encode(src)
    x = model.tgt_embedding(tgt_input)

    tgt_mask = causal_mask(tgt_input.size(1), device=device)
    tgt_padding = padding_mask(tgt_input, pad_idx)
    source_padding = padding_mask(src, pad_idx)

    layers = list(model.decoder.layers)
    if not layers:
        raise ValueError("The decoder has no layers to visualize.")
    target_index = layer_index if layer_index >= 0 else len(layers) + layer_index
    if not 0 <= target_index < len(layers):
        raise IndexError(
            f"layer index {layer_index} is out of range for {len(layers)} layers."
        )

    self_weights = None
    cross_weights = None

    for index, layer in enumerate(layers):
        self_output, self_attention = layer.self_attn(
            x,
            x,
            x,
            key_padding_mask=tgt_padding,
            attn_mask=tgt_mask,
            need_weights=True,
        )
        x = layer.sublayer[0].norm(
            x + layer.sublayer[0].dropout(self_output)
        )

        cross_output, cross_attention = layer.cross_attn(
            x,
            memory,
            memory,
            key_padding_mask=source_padding,
            need_weights=True,
        )
        x = layer.sublayer[1].norm(
            x + layer.sublayer[1].dropout(cross_output)
        )
        x = layer.sublayer[2](x, layer.feed_forward)

        if index == target_index:
            self_weights = self_attention
            cross_weights = cross_attention

    if self_weights is None or cross_weights is None:
        raise RuntimeError("Failed to collect attention weights.")
    return self_weights, cross_weights


def plot_attention(
    self_weights: torch.Tensor,
    cross_weights: torch.Tensor,
    target_tokens: list[str],
    source_tokens: list[str],
    output_path: Path,
    layer_index: int,
) -> None:
    self_map = self_weights[0].detach().cpu().numpy()
    cross_map = cross_weights[0].detach().cpu().numpy()

    figure, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    panels = (
        (
            axes[0],
            self_map,
            target_tokens,
            target_tokens,
            "Decoder self-attention",
            "Key position",
            "Query position",
            "viridis",
        ),
        (
            axes[1],
            cross_map,
            source_tokens,
            target_tokens,
            "Decoder cross-attention",
            "Source position",
            "Target position",
            "magma",
        ),
    )

    for (
        axis,
        matrix,
        x_labels,
        y_labels,
        title,
        x_label,
        y_label,
        cmap,
    ) in panels:
        image = axis.imshow(matrix, aspect="auto", cmap=cmap)
        axis.set_title(title)
        axis.set_xlabel(x_label)
        axis.set_ylabel(y_label)
        axis.set_xticks(range(len(x_labels)))
        axis.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=8)
        axis.set_yticks(range(len(y_labels)))
        axis.set_yticklabels(y_labels, fontsize=8)
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

    figure.suptitle(f"Transformer attention (decoder layer {layer_index})")
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.steps < 1:
        raise ValueError(f"steps must be positive; got {args.steps}.")

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device)
    vocab = build_toy_vocab(args.num_tokens)

    train_dataset = ToySeq2SeqDataset(
        vocab,
        num_samples=args.num_samples,
        min_len=args.min_len,
        max_len=args.max_len,
        reverse=args.task == "reverse",
        seed=args.seed,
    )
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=train_dataset.collate_fn,
        generator=generator,
    )

    model = Transformer(
        src_vocab_size=len(vocab),
        tgt_vocab_size=len(vocab),
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        max_len=args.max_len + 2,
        positional_encoding="sinusoidal",
        share_embeddings=True,
        pad_idx=vocab.pad_idx,
    ).to(device)

    criterion = LabelSmoothingCrossEntropy(
        smoothing=args.label_smoothing,
        ignore_index=vocab.pad_idx,
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1.0,  # overwritten by NoamScheduler
        betas=(0.9, 0.98),
        eps=1e-9,
    )
    scheduler = NoamScheduler(
        optimizer,
        d_model=args.d_model,
        warmup_steps=args.warmup_steps,
    )

    losses = train_steps(
        model,
        train_loader,
        optimizer,
        scheduler,
        criterion,
        steps=args.steps,
        device=device,
    )
    print(
        f"trained {len(losses)} steps | final loss={losses[-1]:.4f} | "
        f"mean loss={sum(losses) / len(losses):.4f}"
    )

    model.eval()
    valid_dataset = ToySeq2SeqDataset(
        vocab,
        num_samples=args.num_samples,
        min_len=args.min_len,
        max_len=args.max_len,
        reverse=args.task == "reverse",
        seed=args.seed + 1,
    )
    sample_index = args.sample_index % len(valid_dataset)
    src, tgt_input, _ = valid_dataset[sample_index]
    src = src.unsqueeze(0).to(device)
    tgt_input = tgt_input.unsqueeze(0).to(device)

    self_weights, cross_weights = extract_attention(
        model,
        src,
        tgt_input,
        pad_idx=vocab.pad_idx,
        layer_index=args.layer,
    )

    source_tokens = [vocab.id_to_token(int(token)) for token in src[0].tolist()]
    target_tokens = [
        vocab.id_to_token(int(token)) for token in tgt_input[0].tolist()
    ]
    output_path = Path(args.output)
    plot_attention(
        self_weights,
        cross_weights,
        target_tokens=target_tokens,
        source_tokens=source_tokens,
        output_path=output_path,
        layer_index=args.layer,
    )
    print(f"source tokens: {source_tokens}")
    print(f"target tokens: {target_tokens}")
    print(f"attention heatmap written to {output_path.resolve()}")


if __name__ == "__main__":
    main()
