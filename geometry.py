from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from PIL import Image, ImageFilter, ImageOps


@dataclass(frozen=True)
class EncoderConfig:
    num_nails: int = 256
    canvas_size: int = 64
    max_lines: int = 240
    min_lines: int = 240
    min_nail_gap: int = 6
    line_radius: int = 1
    line_strength: float = 0.03
    length_power: float = 0.10
    stop_score: float = 0.0
    false_positive_weight: float = 0.20
    recent_edges: int = 80
    blur_radius: float = 0.85
    gamma: float = 0.80
    start_nail: int = 0


def nail_coordinates(num_nails: int, canvas_size: int, integer: bool = False) -> tuple[np.ndarray, np.ndarray]:
    center = (canvas_size - 1) / 2.0
    radius = center - 1.5
    angles = np.linspace(0.0, 2.0 * np.pi, num_nails, endpoint=False)
    x = center + radius * np.cos(angles)
    y = center + radius * np.sin(angles)
    if integer:
        x = np.clip(np.rint(x), 0, canvas_size - 1).astype(np.int32)
        y = np.clip(np.rint(y), 0, canvas_size - 1).astype(np.int32)
    return x, y


def bresenham_points(x0: int, y0: int, x1: int, y1: int) -> list[tuple[int, int]]:
    points: list[tuple[int, int]] = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    error = dx - dy
    while True:
        points.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        doubled = 2 * error
        if doubled > -dy:
            error -= dy
            x0 += sx
        if doubled < dx:
            error += dx
            y0 += sy
    return points


def _thick_line_indices(points: list[tuple[int, int]], canvas_size: int, radius: int) -> list[int]:
    pixels: set[int] = set()
    radius_sq = radius * radius
    for x, y in points:
        for oy in range(-radius, radius + 1):
            for ox in range(-radius, radius + 1):
                if ox * ox + oy * oy > radius_sq:
                    continue
                px = x + ox
                py = y + oy
                if 0 <= px < canvas_size and 0 <= py < canvas_size:
                    pixels.add(py * canvas_size + px)
    return sorted(pixels)


def build_path_table(num_nails: int, canvas_size: int, line_radius: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nail_x, nail_y = nail_coordinates(num_nails, canvas_size, integer=True)
    paths: list[list[list[int]]] = []
    max_length = 1
    for source in range(num_nails):
        source_paths: list[list[int]] = []
        for target in range(num_nails):
            center_line = bresenham_points(
                int(nail_x[source]), int(nail_y[source]),
                int(nail_x[target]), int(nail_y[target]),
            )
            flattened = _thick_line_indices(center_line, canvas_size, line_radius)
            source_paths.append(flattened)
            max_length = max(max_length, len(flattened))
        paths.append(source_paths)

    indices = np.zeros((num_nails, num_nails, max_length), dtype=np.int32)
    mask = np.zeros_like(indices, dtype=np.bool_)
    lengths = np.zeros((num_nails, num_nails), dtype=np.int16)
    for source in range(num_nails):
        for target in range(num_nails):
            path = paths[source][target]
            length = len(path)
            indices[source, target, :length] = path
            mask[source, target, :length] = True
            lengths[source, target] = length
    return indices, mask, lengths


def circular_distance(a: int, b: np.ndarray, num_nails: int) -> np.ndarray:
    direct = np.abs(b - a)
    return np.minimum(direct, num_nails - direct)


def image_to_target(image: Image.Image, config: EncoderConfig) -> np.ndarray:
    prepared = image.convert("L").resize(
        (config.canvas_size, config.canvas_size), Image.Resampling.LANCZOS
    )
    prepared = ImageOps.autocontrast(prepared)
    if config.blur_radius > 0:
        prepared = prepared.filter(ImageFilter.GaussianBlur(config.blur_radius))
    target = np.asarray(prepared, dtype=np.float32) / 255.0
    target = np.power(np.clip(target, 0.0, 1.0), config.gamma)
    return target


def _weighted_squared_error(target: np.ndarray, reconstruction: np.ndarray, false_positive_weight: float) -> np.ndarray:
    missing = np.maximum(target - reconstruction, 0.0)
    excess = np.maximum(reconstruction - target, 0.0)
    return missing * missing + false_positive_weight * excess * excess


def encode_image(
    image: Image.Image,
    config: EncoderConfig,
    path_indices: np.ndarray,
    path_mask: np.ndarray,
    path_lengths: np.ndarray,
) -> tuple[list[int], dict[str, float]]:
    if path_indices.shape[0] != config.num_nails:
        raise ValueError("Path table does not match num_nails")

    target = image_to_target(image, config)
    target_flat = target.reshape(-1)
    reconstruction_flat = np.zeros_like(target_flat)

    current = int(config.start_nail) % config.num_nails
    nails = [current]
    recent = deque(maxlen=max(0, config.recent_edges))
    candidate_ids = np.arange(config.num_nails)
    initial_error = float(_weighted_squared_error(
        target_flat, reconstruction_flat, config.false_positive_weight
    ).sum())
    last_best_score = 0.0

    for _ in range(config.max_lines):
        indices = path_indices[current]
        masks = path_mask[current]
        lengths = path_lengths[current].astype(np.float32)

        target_values = target_flat[indices]
        before_reconstruction = reconstruction_flat[indices]
        after_reconstruction = np.minimum(
            before_reconstruction + config.line_strength, 1.0
        )

        before_error = _weighted_squared_error(
            target_values, before_reconstruction, config.false_positive_weight
        )
        after_error = _weighted_squared_error(
            target_values, after_reconstruction, config.false_positive_weight
        )
        gain = ((before_error - after_error) * masks).sum(axis=1)
        scores = gain / np.maximum(lengths, 1.0) ** config.length_power

        gap = circular_distance(current, candidate_ids, config.num_nails)
        scores[gap < config.min_nail_gap] = -np.inf
        scores[current] = -np.inf

        for edge_a, edge_b in recent:
            if edge_a == current:
                scores[edge_b] = -np.inf
            elif edge_b == current:
                scores[edge_a] = -np.inf

        if len(nails) >= 2:
            scores[nails[-2]] = -np.inf

        best_next = int(np.argmax(scores))
        best_score = float(scores[best_next])
        last_best_score = best_score

        if len(nails) - 1 >= config.min_lines and (
            not np.isfinite(best_score) or best_score < config.stop_score
        ):
            break

        chosen = path_indices[current, best_next]
        chosen_mask = path_mask[current, best_next]
        chosen_pixels = chosen[chosen_mask]
        reconstruction_flat[chosen_pixels] = np.minimum(
            reconstruction_flat[chosen_pixels] + config.line_strength, 1.0
        )

        recent.append((current, best_next))
        nails.append(best_next)
        current = best_next

    final_error = float(_weighted_squared_error(
        target_flat, reconstruction_flat, config.false_positive_weight
    ).sum())
    improvement = 0.0 if initial_error <= 1e-8 else 1.0 - final_error / initial_error

    return nails, {
        "lines": float(max(0, len(nails) - 1)),
        "explained_energy": float(np.clip(improvement, 0.0, 1.0)),
        "last_score": last_best_score,
        "final_error": final_error,
    }


def segments_from_nails(nails: Iterable[int], num_nails: int, canvas_size: int = 64) -> np.ndarray:
    nail_list = list(nails)
    if len(nail_list) < 2:
        return np.empty((0, 2, 2), dtype=np.float32)
    x, y = nail_coordinates(num_nails, canvas_size, integer=False)
    segments = [
        [[x[a], y[a]], [x[b], y[b]]]
        for a, b in zip(nail_list[:-1], nail_list[1:])
    ]
    return np.asarray(segments, dtype=np.float32)
