from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
from tqdm import tqdm
from torchvision.datasets import MNIST

from geometry import EncoderConfig, build_path_table, encode_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert MNIST digits into continuous string-art sequences."
    )
    parser.add_argument("--output", type=Path, default=Path("string_dataset.npz"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--nails", type=int, default=256)
    parser.add_argument("--canvas-size", type=int, default=64)
    parser.add_argument("--max-lines", type=int, default=180)
    parser.add_argument("--min-lines", type=int, default=60)
    parser.add_argument("--min-nail-gap", type=int, default=7)
    parser.add_argument("--erase-strength", type=float, default=0.32)
    parser.add_argument("--length-power", type=float, default=0.55)
    parser.add_argument("--stop-score", type=float, default=0.12)
    parser.add_argument("--recent-edges", type=int, default=24)
    parser.add_argument("--gamma", type=float, default=1.15)
    parser.add_argument("--start-nail", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def balanced_indices(targets: np.ndarray, count: int, seed: int) -> list[int]:
    if count <= 0:
        raise ValueError("--samples must be positive")
    if count > len(targets):
        raise ValueError(
            f"Requested {count} samples, but MNIST train contains {len(targets)}"
        )

    rng = random.Random(seed)
    per_class = count // 10
    remainder = count % 10
    selected: list[int] = []

    for digit in range(10):
        candidates = np.flatnonzero(targets == digit).tolist()
        rng.shuffle(candidates)
        take = per_class + (1 if digit < remainder else 0)
        selected.extend(candidates[:take])

    rng.shuffle(selected)
    return selected


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    random.seed(args.seed)

    config = EncoderConfig(
        num_nails=args.nails,
        canvas_size=args.canvas_size,
        max_lines=args.max_lines,
        min_lines=args.min_lines,
        min_nail_gap=args.min_nail_gap,
        erase_strength=args.erase_strength,
        length_power=args.length_power,
        stop_score=args.stop_score,
        recent_edges=args.recent_edges,
        gamma=args.gamma,
        start_nail=args.start_nail,
    )

    print("Downloading/loading MNIST...")
    dataset = MNIST(root=args.data_dir, train=True, download=True)
    targets = np.asarray(dataset.targets, dtype=np.int64)
    indices = balanced_indices(targets, args.samples, args.seed)

    print("Precomputing all nail-to-nail pixel paths...")
    path_indices, path_mask, path_lengths = build_path_table(
        config.num_nails,
        config.canvas_size,
    )

    eos_token = config.num_nails
    pad_token = config.num_nails + 1
    max_token_count = config.max_lines + 2  # start + endpoints + EOS

    tokens = np.full(
        (len(indices), max_token_count),
        pad_token,
        dtype=np.int16,
    )
    labels = np.empty(len(indices), dtype=np.int8)
    lengths = np.empty(len(indices), dtype=np.int16)
    explained = np.empty(len(indices), dtype=np.float32)

    progress = tqdm(indices, desc="Vectorizing MNIST")
    for row, dataset_index in enumerate(progress):
        image, label = dataset[dataset_index]
        nails, diagnostics = encode_image(
            image,
            config,
            path_indices,
            path_mask,
            path_lengths,
        )

        sequence = nails + [eos_token]
        if len(sequence) > max_token_count:
            sequence = sequence[:max_token_count]
            sequence[-1] = eos_token

        tokens[row, : len(sequence)] = np.asarray(sequence, dtype=np.int16)
        labels[row] = int(label)
        lengths[row] = len(sequence)
        explained[row] = diagnostics["explained_energy"]

        if row % 25 == 0:
            progress.set_postfix(
                lines=int(diagnostics["lines"]),
                explained=f"{diagnostics['explained_energy']:.1%}",
            )

    metadata = {
        "format_version": 2,
        "num_samples": len(indices),
        "num_nails": config.num_nails,
        "num_classes": 10,
        "canvas_size": config.canvas_size,
        "max_lines": config.max_lines,
        "min_lines": config.min_lines,
        "min_nail_gap": config.min_nail_gap,
        "erase_strength": config.erase_strength,
        "length_power": config.length_power,
        "stop_score": config.stop_score,
        "recent_edges": config.recent_edges,
        "gamma": config.gamma,
        "start_nail": config.start_nail,
        "eos_token": eos_token,
        "pad_token": pad_token,
        "class_token_offset": config.num_nails + 2,
        "seed": args.seed,
        "mean_explained_energy": float(explained.mean()),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        tokens=tokens,
        labels=labels,
        lengths=lengths,
        explained_energy=explained,
        metadata=np.asarray(json.dumps(metadata)),
    )

    print(f"\nSaved: {args.output}")
    print(f"Samples: {len(indices)}")
    print(f"Mean encoded lines: {float((lengths - 2).mean()):.1f}")
    print(f"Mean explained energy: {float(explained.mean()):.1%}")


if __name__ == "__main__":
    main()
