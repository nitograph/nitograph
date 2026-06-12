from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

from geometry import segments_from_nails


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preview encoded dataset rows.")
    parser.add_argument("--dataset", type=Path, default=Path("string_dataset.npz"))
    parser.add_argument("--output", type=Path, default=Path("dataset_preview.png"))
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    archive = np.load(args.dataset, allow_pickle=False)
    tokens = archive["tokens"]
    labels = archive["labels"]
    metadata = json.loads(str(archive["metadata"].item()))

    num_nails = int(metadata["num_nails"])
    eos_token = int(metadata["eos_token"])
    pad_token = int(metadata["pad_token"])
    canvas_size = int(metadata["canvas_size"])

    rng = np.random.default_rng(args.seed)
    count = min(args.count, len(tokens))
    rows = rng.choice(len(tokens), size=count, replace=False)

    columns = 5
    grid_rows = int(np.ceil(count / columns))
    figure, axes = plt.subplots(
        grid_rows,
        columns,
        figsize=(columns * 2.4, grid_rows * 2.4),
        facecolor="black",
    )
    axes = np.asarray(axes).reshape(-1)

    for axis, row in zip(axes, rows):
        sequence = [
            int(token)
            for token in tokens[row]
            if token not in (eos_token, pad_token)
        ]
        segments = segments_from_nails(
            sequence,
            num_nails,
            canvas_size,
        )

        axis.set_facecolor("black")
        axis.add_collection(
            LineCollection(
                segments,
                colors=[(0.0, 1.0, 1.0, 0.18)],
                linewidths=0.7,
            )
        )
        axis.set_xlim(0, canvas_size)
        axis.set_ylim(canvas_size, 0)
        axis.set_aspect("equal")
        axis.axis("off")
        axis.set_title(str(int(labels[row])), color="white")

    for axis in axes[count:]:
        axis.axis("off")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        args.output,
        dpi=180,
        facecolor="black",
        bbox_inches="tight",
    )
    plt.close(figure)
    print(f"Saved preview: {args.output}")


if __name__ == "__main__":
    main()
