# Author: [SunMuzihang] | Student ID: [2260031490]

"""Minimal CPU training loop for the from-scratch Transformer.

Run from the repository root:

    python scripts/train.py --steps 10

The default task is a tiny synthetic copy/reverse sequence-to-sequence problem,
so no external dataset or download is required.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Allow `python scripts/train.py` from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data import ToySeq2SeqDataset, build_toy_vocab
from model import Transformer
from training import (
    LabelSmoothingCrossEntropy,
    NoamScheduler,
    load_checkpoint,
    save_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--steps",
        type=int,
        default=10,
        help="Number of optimizer steps; 0 means one full epoch (default: 10).",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-samples", type=int, default=128)
    parser.add_argument("--min-len", type=int, default=3)
    parser.add_argument("--max-len", type=int, default=8)
    parser.add_argument("--num-tokens", type=int, default=32)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-encoder-layers", type=int, default=2)
    parser.add_argument("--num-decoder-layers", type=int, default=2)
    parser.add_argument("--d-ff", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=100,
        help="Warmup steps for this toy run (paper base model uses 4000).",
    )
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--lr-factor", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional path to save the final training checkpoint.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Optional checkpoint path to restore before training.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.steps < 0:
        raise ValueError(f"steps must be non-negative; got {args.steps}.")
    if args.batch_size < 1:
        raise ValueError(f"batch_size must be positive; got {args.batch_size}.")
    if args.num_samples < 1:
        raise ValueError(f"num_samples must be positive; got {args.num_samples}.")

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device)

    vocab = build_toy_vocab(args.num_tokens)
    dataset = ToySeq2SeqDataset(
        vocab,
        num_samples=args.num_samples,
        min_len=args.min_len,
        max_len=args.max_len,
        reverse=True,
        seed=args.seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=dataset.collate_fn,
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
        factor=args.lr_factor,
    )

    if args.resume:
        checkpoint = load_checkpoint(
            args.resume,
            model,
            optimizer,
            scheduler,
            map_location=device,
        )
        print(
            f"resumed from {args.resume} "
            f"(completed steps={checkpoint.get('step', 'unknown')})"
        )

    n_parameters = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"device={device} | vocab={len(vocab)} | params={n_parameters:,} | "
        f"batches={len(loader)}"
    )

    model.train()
    losses: list[float] = []
    for batch_index, (src, tgt_input, tgt_output) in enumerate(loader, start=1):
        src = src.to(device)
        tgt_input = tgt_input.to(device)
        tgt_output = tgt_output.to(device)

        current_step = scheduler.step_num
        current_lr = float(optimizer.param_groups[0]["lr"])
        optimizer.zero_grad(set_to_none=True)
        logits = model(src, tgt_input)
        loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            tgt_output.reshape(-1),
        )
        loss.backward()
        optimizer.step()
        scheduler.step()

        loss_value = float(loss.item())
        losses.append(loss_value)
        print(f"step {current_step:02d} | loss {loss_value:.4f} | lr {current_lr:.6f}")

        if args.steps and batch_index >= args.steps:
            break

    if not losses:
        raise RuntimeError("No training steps were executed.")

    mean_loss = sum(losses) / len(losses)
    print(
        f"done | steps={len(losses)} | mean loss={mean_loss:.4f} | "
        f"final loss={losses[-1]:.4f}"
    )

    if args.checkpoint:
        saved_path = save_checkpoint(
            args.checkpoint,
            model,
            optimizer,
            scheduler,
            step=max(0, scheduler.step_num - 1),
            extra={"args": vars(args)},
        )
        print(f"checkpoint saved to {saved_path}")


if __name__ == "__main__":
    main()
