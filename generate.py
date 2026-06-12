from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.collections import LineCollection

from geometry import nail_coordinates, segments_from_nails
from model import MiniCausalTransformer, config_from_dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate and render a neon string-art digit."
    )
    parser.add_argument("digit", type=int, choices=range(10))
    parser.add_argument("--checkpoint", type=Path, default=Path("string_ai.pt"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--sequence-output", type=Path, default=None)
    parser.add_argument("--temperature", type=float, default=0.85)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--repetition-penalty", type=float, default=1.12)
    parser.add_argument("--min-lines", type=int, default=50)
    parser.add_argument("--max-lines", type=int, default=None)
    parser.add_argument("--min-nail-gap", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dpi", type=int, default=512)
    parser.add_argument("--size-inches", type=float, default=8.0)
    parser.add_argument("--show", action="store_true")
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "mps", "cpu"),
        default="auto",
    )
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def circular_gap(a: int, b: int, count: int) -> int:
    distance = abs(a - b)
    return min(distance, count - distance)


def apply_repetition_penalty(
    logits: torch.Tensor,
    generated: list[int],
    penalty: float,
) -> None:
    if penalty <= 1.0:
        return
    for token in set(generated[-64:]):
        value = logits[token]
        logits[token] = value / penalty if value >= 0 else value * penalty


def sample_token(
    logits: torch.Tensor,
    temperature: float,
    top_k: int,
    generator: torch.Generator,
) -> int:
    if temperature <= 0.0:
        return int(torch.argmax(logits).item())

    logits = logits / temperature
    if 0 < top_k < logits.numel():
        threshold = torch.topk(logits, top_k).values[-1]
        logits = torch.where(
            logits < threshold,
            torch.full_like(logits, -torch.inf),
            logits,
        )

    probabilities = torch.softmax(logits, dim=-1)
    if not torch.isfinite(probabilities).all() or probabilities.sum() <= 0:
        return int(torch.argmax(logits).item())

    # MPS currently samples most reliably on CPU; CUDA/CPU use a matching
    # device generator directly.
    sample_probabilities = (
        probabilities.cpu()
        if probabilities.device.type == "mps"
        else probabilities
    )
    return int(
        torch.multinomial(
            sample_probabilities,
            num_samples=1,
            generator=generator,
        ).item()
    )


@torch.no_grad()
def generate_nails(
    model: MiniCausalTransformer,
    digit: int,
    max_lines: int,
    min_lines: int,
    min_nail_gap: int,
    temperature: float,
    top_k: int,
    repetition_penalty: float,
    seed: int,
    device: torch.device,
) -> list[int]:
    config = model.config
    class_token = config.class_token_offset + digit
    context = torch.tensor([[class_token]], dtype=torch.long, device=device)
    nails: list[int] = []

    # A CPU generator makes seeded sampling reproducible across CPU runs.
    # CUDA uses its own generator because torch.multinomial requires matching
    # generator/device types.
    generator_device = device.type if device.type in {"cpu", "cuda"} else "cpu"
    generator = torch.Generator(device=generator_device).manual_seed(seed)

    # max_lines chords require max_lines + 1 nail positions.
    max_nails = min(max_lines + 1, config.max_seq_len)

    while len(nails) < max_nails:
        logits = model(context)[:, -1, :].squeeze(0).float()

        apply_repetition_penalty(logits, nails, repetition_penalty)

        if len(nails) - 1 < min_lines:
            logits[config.eos_token] = -torch.inf

        if nails:
            current = nails[-1]
            logits[current] = -torch.inf

            for candidate in range(config.num_nails):
                if circular_gap(current, candidate, config.num_nails) < min_nail_gap:
                    logits[candidate] = -torch.inf

            # Prevent immediate A -> B -> A reversal.
            if len(nails) >= 2:
                logits[nails[-2]] = -torch.inf

        token = sample_token(
            logits,
            temperature=temperature,
            top_k=top_k,
            generator=generator,
        )

        if token == config.eos_token:
            break

        nails.append(token)
        context = torch.cat(
            [
                context,
                torch.tensor([[token]], dtype=torch.long, device=device),
            ],
            dim=1,
        )

        # No later forward pass may exceed the trained positional range.
        if context.shape[1] >= config.max_seq_len:
            break

    return nails


def render_4k(
    nails: list[int],
    num_nails: int,
    canvas_size: int,
    output: Path,
    dpi: int,
    size_inches: float,
    digit: int,
) -> None:
    segments = segments_from_nails(nails, num_nails, canvas_size)

    figure = plt.figure(
        figsize=(size_inches, size_inches),
        dpi=dpi,
        facecolor="black",
    )
    axis = figure.add_axes([0.0, 0.0, 1.0, 1.0])
    axis.set_facecolor("black")
    axis.set_xlim(0, canvas_size)
    axis.set_ylim(canvas_size, 0)
    axis.set_aspect("equal")
    axis.axis("off")

    # Three batched line layers create a glow without plotting each line
    # separately. Rendering is CPU-based; generation can use CUDA/MPS.
    glow_layers = (
        (9.0, 0.025),
        (4.0, 0.075),
        (1.1, 0.42),
    )
    for width, alpha in glow_layers:
        collection = LineCollection(
            segments,
            colors=[(0.0, 1.0, 1.0, alpha)],
            linewidths=width,
            capstyle="round",
            joinstyle="round",
        )
        axis.add_collection(collection)

    # Small nail ring gives the output a physical string-art feel.
    nail_x, nail_y = nail_coordinates(num_nails, canvas_size, integer=False)
    axis.scatter(
        nail_x,
        nail_y,
        s=1.4,
        c=[(0.75, 1.0, 1.0, 0.32)],
        linewidths=0,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output,
        dpi=dpi,
        facecolor=figure.get_facecolor(),
        edgecolor="none",
        pad_inches=0,
    )
    plt.close(figure)

    pixels = round(size_inches * dpi)
    print(f"Rendered digit {digit}: {pixels}x{pixels}px -> {output}")


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = choose_device(args.device)
    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=True,
    )
    config = config_from_dict(checkpoint["model_config"])
    model = MiniCausalTransformer(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    data_metadata = checkpoint.get("data_metadata", {})
    canvas_size = int(data_metadata.get("canvas_size", 64))
    trained_max_lines = int(
        data_metadata.get("max_lines", config.max_seq_len - 2)
    )
    max_lines = (
        min(args.max_lines, trained_max_lines)
        if args.max_lines is not None
        else trained_max_lines
    )
    min_nail_gap = (
        args.min_nail_gap
        if args.min_nail_gap is not None
        else int(data_metadata.get("min_nail_gap", 7))
    )

    print(f"Device: {device}")
    print(f"Generating digit {args.digit}...")
    nails = generate_nails(
        model=model,
        digit=args.digit,
        max_lines=max_lines,
        min_lines=min(args.min_lines, max_lines),
        min_nail_gap=min_nail_gap,
        temperature=args.temperature,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        seed=args.seed,
        device=device,
    )

    if len(nails) < 2:
        raise RuntimeError(
            "The model produced fewer than two nails. Train longer or use "
            "--temperature 0.7 --top-k 16."
        )

    output = args.output or Path(f"digit_{args.digit}_4k.png")
    sequence_output = (
        args.sequence_output
        or output.with_suffix(".json")
    )

    render_4k(
        nails=nails,
        num_nails=config.num_nails,
        canvas_size=canvas_size,
        output=output,
        dpi=args.dpi,
        size_inches=args.size_inches,
        digit=args.digit,
    )

    sequence_payload = {
        "digit": args.digit,
        "num_nails": config.num_nails,
        "line_count": len(nails) - 1,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "seed": args.seed,
        "nails": nails,
    }
    sequence_output.parent.mkdir(parents=True, exist_ok=True)
    sequence_output.write_text(
        json.dumps(sequence_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Sequence: {sequence_output}")

    if args.show:
        image = plt.imread(output)
        plt.figure(figsize=(8, 8))
        plt.imshow(image)
        plt.axis("off")
        plt.show()


if __name__ == "__main__":
    main()
