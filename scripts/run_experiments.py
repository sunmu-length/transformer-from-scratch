# Author: [SunMuzihang] | Student ID: [2260031490]

"""Run the Week 4 ablations on the toy reverse task.

Example:
    python scripts/run_experiments.py --steps 300
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data import ToySeq2SeqDataset, build_toy_vocab
from model import Transformer
from training import (
    LabelSmoothingCrossEntropy,
    NoamScheduler,
    evaluate,
    train_steps,
)


@dataclass(frozen=True)
class Experiment:
    name: str
    num_heads: int
    positional_encoding: str


EXPERIMENTS = (
    Experiment("base", num_heads=4, positional_encoding="sinusoidal"),
    Experiment("single_head", num_heads=1, positional_encoding="sinusoidal"),
    Experiment("learned_pe", num_heads=4, positional_encoding="learned"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument(
        "--task",
        choices=("reverse", "copy"),
        default="reverse",
        help="Toy task: reverse the source or copy it.",
    )
    parser.add_argument(
        "--configs",
        type=str,
        default="base,single_head,learned_pe",
        help="Comma-separated experiment names.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--train-samples", type=int, default=2048)
    parser.add_argument("--valid-samples", type=int, default=256)
    parser.add_argument("--min-len", type=int, default=3)
    parser.add_argument("--max-len", type=int, default=8)
    parser.add_argument("--num-tokens", type=int, default=32)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--num-encoder-layers", type=int, default=2)
    parser.add_argument("--num-decoder-layers", type=int, default=2)
    parser.add_argument("--d-ff", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def make_loaders(
    args: argparse.Namespace,
    vocab,
) -> tuple[DataLoader, DataLoader]:
    train_dataset = ToySeq2SeqDataset(
        vocab,
        num_samples=args.train_samples,
        min_len=args.min_len,
        max_len=args.max_len,
        reverse=args.task == "reverse",
        seed=args.seed,
    )
    valid_dataset = ToySeq2SeqDataset(
        vocab,
        num_samples=args.valid_samples,
        min_len=args.min_len,
        max_len=args.max_len,
        reverse=args.task == "reverse",
        seed=args.seed + 1,
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
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=valid_dataset.collate_fn,
    )
    return train_loader, valid_loader


def run_experiment(
    experiment: Experiment,
    args: argparse.Namespace,
    vocab,
) -> dict:
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device)
    train_loader, valid_loader = make_loaders(args, vocab)

    model = Transformer(
        src_vocab_size=len(vocab),
        tgt_vocab_size=len(vocab),
        d_model=args.d_model,
        num_heads=experiment.num_heads,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        max_len=args.max_len + 2,
        positional_encoding=experiment.positional_encoding,
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
    metrics = evaluate(
        model,
        valid_loader,
        pad_idx=vocab.pad_idx,
        device=device,
    )

    return {
        "name": experiment.name,
        "num_heads": experiment.num_heads,
        "positional_encoding": experiment.positional_encoding,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "final_train_loss": losses[-1],
        "mean_train_loss": sum(losses) / len(losses),
        "losses": losses,
        **metrics,
    }


def moving_average(values: list[float], window: int = 15) -> list[float]:
    if window <= 1:
        return list(values)
    smoothed = []
    for index in range(len(values)):
        start = max(0, index - window + 1)
        chunk = values[start : index + 1]
        smoothed.append(sum(chunk) / len(chunk))
    return smoothed


def save_results(results: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "ablation.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    max_steps = max(len(result["losses"]) for result in results)
    with (output_dir / "loss_curves.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["step", *[result["name"] for result in results]])
        for step in range(1, max_steps + 1):
            row = [step]
            for result in results:
                losses = result["losses"]
                row.append(losses[step - 1] if step <= len(losses) else "")
            writer.writerow(row)

    with (output_dir / "ablation_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "config",
                "num_heads",
                "positional_encoding",
                "parameters",
                "val_loss",
                "token_accuracy",
                "exact_match",
                "final_train_loss",
                "mean_train_loss",
            ]
        )
        for result in results:
            writer.writerow(
                [
                    result["name"],
                    result["num_heads"],
                    result["positional_encoding"],
                    result["parameters"],
                    f"{result['loss']:.6f}",
                    f"{result['token_accuracy']:.6f}",
                    f"{result['exact_match']:.6f}",
                    f"{result['final_train_loss']:.6f}",
                    f"{result['mean_train_loss']:.6f}",
                ]
            )


def plot_loss_curves(results: list[dict], output_dir: Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 5))
    for result in results:
        steps = range(1, len(result["losses"]) + 1)
        axis.plot(steps, result["losses"], alpha=0.2, linewidth=1)
        axis.plot(
            steps,
            moving_average(result["losses"]),
            linewidth=2,
            label=result["name"],
        )

    axis.set_xlabel("Training step")
    axis.set_ylabel("Label-smoothed loss")
    axis.set_title("Training loss on the toy reverse task")
    axis.legend()
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_dir / "loss_curves.png", dpi=150)
    plt.close(figure)


def plot_metrics(results: list[dict], output_dir: Path) -> None:
    names = [result["name"] for result in results]
    positions = list(range(len(names)))
    width = 0.38

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.bar(
        [position - width / 2 for position in positions],
        [result["token_accuracy"] for result in results],
        width,
        label="Token accuracy",
    )
    axis.bar(
        [position + width / 2 for position in positions],
        [result["exact_match"] for result in results],
        width,
        label="Exact match",
    )
    axis.set_xticks(positions)
    axis.set_xticklabels(names)
    axis.set_ylim(0.0, 1.05)
    axis.set_ylabel("Validation score")
    axis.set_title("Validation accuracy on the toy reverse task")
    axis.legend()
    axis.grid(axis="y", alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_dir / "ablation_accuracy.png", dpi=150)
    plt.close(figure)


def print_summary(results: list[dict]) -> None:
    print(
        f"{'config':<14} {'heads':>5} {'PE':<10} {'params':>9} "
        f"{'val loss':>9} {'token acc':>9} {'exact':>7}"
    )
    for result in results:
        print(
            f"{result['name']:<14} {result['num_heads']:>5} "
            f"{result['positional_encoding']:<10} "
            f"{result['parameters']:>9,} {result['loss']:>9.4f} "
            f"{result['token_accuracy']:>9.4f} "
            f"{result['exact_match']:>7.4f}"
        )


def main() -> None:
    args = parse_args()
    if args.steps < 1:
        raise ValueError(f"steps must be positive; got {args.steps}.")
    if args.batch_size < 1:
        raise ValueError(f"batch_size must be positive; got {args.batch_size}.")

    requested = {
        name.strip() for name in args.configs.split(",") if name.strip()
    }
    selected = [
        experiment for experiment in EXPERIMENTS if experiment.name in requested
    ]
    missing = requested - {experiment.name for experiment in selected}
    if missing:
        raise ValueError(f"Unknown experiment(s): {sorted(missing)}.")
    if not selected:
        raise ValueError("No experiments selected.")

    output_dir = Path(args.output_dir)
    vocab = build_toy_vocab(args.num_tokens)
    print(
        f"device={args.device} | task={args.task} | vocab={len(vocab)} | "
        f"train={args.train_samples} | valid={args.valid_samples} | "
        f"steps={args.steps}"
    )

    results = []
    for experiment in selected:
        print(f"\n[{experiment.name}] training...")
        result = run_experiment(experiment, args, vocab)
        results.append(result)
        print(
            f"  val loss={result['loss']:.4f} | "
            f"token acc={result['token_accuracy']:.4f} | "
            f"exact match={result['exact_match']:.4f}"
        )

    save_results(results, output_dir)
    plot_loss_curves(results, output_dir)
    plot_metrics(results, output_dir)
    print("\n" + "=" * 72)
    print_summary(results)
    print(f"\nresults written to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
