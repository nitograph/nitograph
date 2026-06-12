from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision.datasets import MNIST

from geometry import EncoderConfig, build_path_table, encode_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert MNIST digits into precise continuous string-art sequences."
    )
    parser.add_argument("--output", type=Path, default=Path("string_dataset.npz"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument(
        "--mode",
        choices=("prototypes", "examples"),
        default="prototypes",
        help=(
            "prototypes creates canonical class templates and is best when the "
            "model receives only a digit label; examples preserves handwriting "
            "variation but is harder to learn from a class label alone"
        ),
    )
    parser.add_argument("--prototype-count", type=int, default=1)
    parser.add_argument("--prototype-pool", type=int, default=1000)
    parser.add_argument("--nails", type=int, default=256)
    parser.add_argument("--canvas-size", type=int, default=64)
    parser.add_argument("--max-lines", type=int, default=240)
    parser.add_argument("--min-lines", type=int, default=240)
    parser.add_argument("--min-nail-gap", type=int, default=6)
    parser.add_argument("--line-radius", type=int, default=1)
    parser.add_argument("--line-strength", type=float, default=0.03)
    parser.add_argument("--length-power", type=float, default=0.10)
    parser.add_argument("--stop-score", type=float, default=0.0)
    parser.add_argument("--false-positive-weight", type=float, default=0.20)
    parser.add_argument("--recent-edges", type=int, default=80)
    parser.add_argument("--blur-radius", type=float, default=0.85)
    parser.add_argument("--gamma", type=float, default=0.80)
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


def build_prototypes(
    dataset: MNIST,
    targets: np.ndarray,
    prototype_count: int,
    prototype_pool: int,
    seed: int,
) -> list[tuple[Image.Image, int]]:
    if prototype_count <= 0:
        raise ValueError("--prototype-count must be positive")
    if prototype_pool < prototype_count:
        raise ValueError("--prototype-pool must be >= --prototype-count")

    rng = random.Random(seed)
    prototypes: list[tuple[Image.Image, int]] = []

    for digit in range(10):
        candidates = np.flatnonzero(targets == digit).tolist()
        rng.shuffle(candidates)
        selected = candidates[: min(prototype_pool, len(candidates))]
        groups = np.array_split(np.asarray(selected, dtype=np.int64), prototype_count)

        for group in groups:
            if len(group) == 0:
                continue
            arrays = [
                np.asarray(dataset[int(index)][0], dtype=np.float32)
                for index in group
            ]
            average = np.mean(np.stack(arrays, axis=0), axis=0)
            average = np.clip(np.rint(average), 0, 255).astype(np.uint8)
            prototypes.append((Image.fromarray(average, mode="L"), digit))

    return prototypes


def encode_sources(
    sources: list[tuple[Image.Image, int]],
    config: EncoderConfig,
    path_indices: np.ndarray,
    path_mask: np.ndarray,
    path_lengths: np.ndarray,
) -> list[tuple[list[int], int, dict[str, float]]]:
    encoded: list[tuple[list[int], int, dict[str, float]]] = []
    progress = tqdm(sources, desc="Encoding source images")

    for image, label in progress:
        nails, diagnostics = encode_image(
            image,
            config,
            path_indices,
            path_mask,
            path_lengths,
        )
        encoded.append((nails, int(label), diagnostics))
        progress.set_postfix(
            digit=int(label),
            lines=int(diagnostics["lines"]),
            improvement=f"{diagnostics['explained_energy']:.1%}",
        )

    return encoded


def materialize_dataset(
    encoded_sources: list[tuple[list[int], int, dict[str, float]]],
    sample_count: int,
    config: EncoderConfig,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not encoded_sources:
        raise ValueError("No encoded sources were produced")

    eos_token = config.num_nails
    pad_token = config.num_nails + 1
    max_token_count = config.max_lines + 2

    tokens = np.full(
        (sample_count, max_token_count),
        pad_token,
        dtype=np.int16,
    )
    labels = np.empty(sample_count, dtype=np.int8)
    lengths = np.empty(sample_count, dtype=np.int16)
    explained = np.empty(sample_count, dtype=np.float32)

    rng = random.Random(seed)
    order = list(range(len(encoded_sources)))

    for row in range(sample_count):
        if row % len(order) == 0:
            rng.shuffle(order)
        source_index = order[row % len(order)]
        nails, label, diagnostics = encoded_sources[source_index]

        sequence = nails + [eos_token]
        if len(sequence) > max_token_count:
            sequence = sequence[:max_token_count]
            sequence[-1] = eos_token

        tokens[row, : len(sequence)] = np.asarray(sequence, dtype=np.int16)
        labels[row] = label
        lengths[row] = len(sequence)
        explained[row] = diagnostics["explained_energy"]

    permutation = np.asarray(rng.sample(range(sample_count), sample_count))
    return (
        tokens[permutation],
        labels[permutation],
        lengths[permutation],
        explained[permutation],
    )


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
        line_radius=args.line_radius,
        line_strength=args.line_strength,
        length_power=args.length_power,
        stop_score=args.stop_score,
        false_positive_weight=args.false_positive_weight,
        recent_edges=args.recent_edges,
        blur_radius=args.blur_radius,
        gamma=args.gamma,
        start_nail=args.start_nail,
    )

    print("Downloading/loading MNIST...")
    dataset = MNIST(root=args.data_dir, train=True, download=True)
    targets = np.asarray(dataset.targets, dtype=np.int64)

    print("Precomputing thick nail-to-nail line masks...")
    path_indices, path_mask, path_lengths = build_path_table(
        config.num_nails,
        config.canvas_size,
        config.line_radius,
    )

    if args.mode == "prototypes":
        print(
            f"Building {args.prototype_count} canonical prototype(s) per digit "
            f"from up to {args.prototype_pool} MNIST examples..."
        )
        sources = build_prototypes(
            dataset,
            targets,
            args.prototype_count,
            args.prototype_pool,
            args.seed,
        )
    else:
        selected = balanced_indices(targets, args.samples, args.seed)
        sources = [(dataset[index][0], int(targets[index])) for index in selected]

    encoded_sources = encode_sources(
        sources,
        config,
        path_indices,
        path_mask,
        path_lengths,
    )

    materialized_samples = args.samples if args.mode == "prototypes" else len(encoded_sources)
    tokens, labels, lengths, explained = materialize_dataset(
        encoded_sources,
        materialized_samples,
        config,
        args.seed,
    )

    metadata = {
        "format_version": 3,
        "mode": args.mode,
        "num_samples": int(materialized_samples),
        "unique_sources": len(encoded_sources),
        "prototype_count": args.prototype_count if args.mode == "prototypes" else 0,
        "prototype_pool": args.prototype_pool if args.mode == "prototypes" else 0,
        "num_nails": config.num_nails,
        "num_classes": 10,
        "canvas_size": config.canvas_size,
        "max_lines": config.max_lines,
        "min_lines": config.min_lines,
        "min_nail_gap": config.min_nail_gap,
        "line_radius": config.line_radius,
        "line_strength": config.line_strength,
        "length_power": config.length_power,
        "stop_score": config.stop_score,
        "false_positive_weight": config.false_positive_weight,
        "recent_edges": config.recent_edges,
        "blur_radius": config.blur_radius,
        "gamma": config.gamma,
        "start_nail": config.start_nail,
        "eos_token": config.num_nails,
        "pad_token": config.num_nails + 1,
        "class_token_offset": config.num_nails + 2,
        "seed": args.seed,
        "mean_reconstruction_improvement": float(explained.mean()),
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
    print(f"Rows: {len(tokens)}")
    print(f"Unique encoded sources: {len(encoded_sources)}")
    print(f"Mean encoded lines: {float((lengths - 2).mean()):.1f}")
    print(f"Mean reconstruction improvement: {float(explained.mean()):.1%}")


if __name__ == "__main__":
    main()
