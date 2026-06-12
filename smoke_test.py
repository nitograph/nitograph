from __future__ import annotations

import numpy as np
import torch
from PIL import Image, ImageDraw

from geometry import EncoderConfig, build_path_table, encode_image
from model import MiniCausalTransformer, ModelConfig


def main() -> None:
    config = EncoderConfig(
        num_nails=32,
        canvas_size=24,
        max_lines=12,
        min_lines=4,
        min_nail_gap=2,
        recent_edges=4,
    )
    paths, masks, lengths = build_path_table(
        config.num_nails,
        config.canvas_size,
    )

    image = Image.new("L", (24, 24), 0)
    draw = ImageDraw.Draw(image)
    draw.line((6, 4, 17, 4, 9, 20), fill=255, width=3)

    nails, diagnostics = encode_image(
        image,
        config,
        paths,
        masks,
        lengths,
    )
    assert len(nails) >= 2
    assert all(0 <= nail < config.num_nails for nail in nails)
    assert 0.0 <= diagnostics["explained_energy"] <= 1.0

    model_config = ModelConfig(
        num_nails=32,
        max_seq_len=14,
        embed_dim=32,
        num_heads=4,
        num_layers=2,
        ff_dim=64,
    )
    model = MiniCausalTransformer(model_config)
    tokens = torch.tensor(
        [[model_config.class_token_offset + 7, 0, 5, 10]],
        dtype=torch.long,
    )
    logits = model(tokens)
    assert logits.shape == (1, 4, model_config.output_vocab_size)
    assert torch.isfinite(logits).all()

    print("Smoke test passed.")
    print(f"Encoded nails: {nails}")
    print(f"Diagnostics: {diagnostics}")


if __name__ == "__main__":
    main()
