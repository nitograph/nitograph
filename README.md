# Nitograph

**Nitograph turns MNIST digits into neon string-art drawings, then trains a small class-conditioned Transformer to generate new continuous nail sequences.**

[Русский](README.ru.md) | [中文](README.zh.md)

![Dataset preview](images/dataset_preview_img.png)

![Generated digit 7](images/digit_7_4k.png)

## Why It Is Interesting

Nitograph is not a prompt-to-image toy. It is a compact generative pipeline for a very specific physical representation: a single thread moving around a circular set of nails.

- Converts MNIST digits into ordered nail-to-nail paths.
- Trains an autoregressive Transformer on tokenized string-art sequences.
- Samples new paths conditioned only on the digit class.
- Renders crisp 4096x4096 neon output with batched Matplotlib line layers.
- Saves the exact generated nail sequence as JSON, so the image is reproducible and portable.

In short: **image -> string-art encoder -> sequence dataset -> causal Transformer -> generated thread path -> 4K render**.

## Demo Assets

This repository includes preview images in [`images/`](images):

- [`images/dataset_preview_img.png`](images/dataset_preview_img.png) shows encoded dataset examples.
- [`images/digit_7_4k.png`](images/digit_7_4k.png) shows a generated 4K output image.

Model weights and a real generated output example are available in [`model_and_output_examples/`](model_and_output_examples):

- [`model_and_output_examples/string_ai.pt`](model_and_output_examples/string_ai.pt) - trained checkpoint.
- [`model_and_output_examples/digit_7_4k.png`](model_and_output_examples/digit_7_4k.png) - generated render.
- [`model_and_output_examples/digit_7_4k.json`](model_and_output_examples/digit_7_4k.json) - exact nail sequence.

## Installation

Python 3.11 or 3.12 is recommended.

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/macOS
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

For NVIDIA GPUs, install the PyTorch build that matches your CUDA version from the official PyTorch installer, then install the remaining dependencies.

## Quick Start

### 1. Generate A Dataset

Fast smoke run:

```bash
python prepare_dataset.py --samples 100 --max-lines 120 --output smoke_dataset.npz
python preview_dataset.py --dataset smoke_dataset.npz --output dataset_preview.png
```

Better training run:

```bash
python prepare_dataset.py --samples 5000 --max-lines 240
python preview_dataset.py --dataset string_dataset.npz --output dataset_preview.png
```

By default, the dataset builder creates canonical prototypes per digit. This works well because the model is conditioned by class label, not by a full source image.

### 2. Train The Transformer

```bash
python train_ai.py --epochs 15 --batch-size 32
```

Low-memory GPU:

```bash
python train_ai.py --epochs 15 --batch-size 8
```

CPU-friendly smaller model:

```bash
python train_ai.py --epochs 10 --batch-size 16 --embed-dim 128 --heads 4 --layers 3 --ff-dim 384
```

Training writes:

- `string_ai.pt` - best validation checkpoint.
- `string_ai_last.pt` - latest checkpoint.

### 3. Generate A 4K Digit

```bash
python generate.py 7
```

This creates:

- `digit_7_4k.png` - 4096x4096 rendered output.
- `digit_7_4k.json` - exact generated nail sequence.

Sampling examples:

```bash
python generate.py 3 --temperature 0.70 --top-k 16 --seed 10
python generate.py 9 --temperature 1.00 --top-k 48 --seed 77
python generate.py 4 --temperature 0 --top-k 1
```

`--temperature 0` enables deterministic greedy decoding. It is reproducible, but sampling usually produces richer thread paths.

## How It Works

### String-Art Encoder

`prepare_dataset.py` downloads MNIST, places nails on a circle, precomputes thick Bresenham nail-to-nail paths, and greedily chooses the next chord that improves reconstruction of the target digit.

The encoder includes practical constraints:

- minimum circular nail gap;
- no immediate `A -> B -> A` reversal;
- recent edge suppression;
- length-normalized scoring;
- false-positive penalty for overdrawn pixels;
- compressed `.npz` dataset output.

### Sequence Model

`train_ai.py` trains a compact causal Transformer:

- separate class tokens for digits `0..9`;
- nail tokens `0..255`;
- explicit `EOS` and `PAD` tokens;
- validation split;
- CUDA AMP when available;
- gradient clipping;
- best-checkpoint saving.

### Renderer

`generate.py` samples a nail sequence with temperature, top-k, and repetition penalty, then renders a 4K neon string-art image. The render uses batched `LineCollection` layers for glow and exports the nail path as JSON.

## Project Structure

```text
nitograph/
├── geometry.py                 # nail geometry, path tables, encoder
├── model.py                    # mini causal Transformer
├── prepare_dataset.py          # MNIST -> tokenized string-art dataset
├── preview_dataset.py          # dataset visual preview
├── train_ai.py                 # training loop
├── generate.py                 # sequence sampling and 4K rendering
├── images/                     # README preview images
├── model_and_output_examples/  # checkpoint and real output example
├── requirements.txt
├── README.md
├── README.ru.md
└── README.zh.md
```

## Reproducibility Notes

- Dataset generation and sampling accept `--seed`.
- Generated JSON files store the digit, nail count, line count, sampling settings, seed, and full nail list.
- Rendering size is controlled by `--size-inches` and `--dpi`; the default `8 * 512` produces 4096x4096 pixels.

## Honest Scope

Nitograph is a focused research/art MVP. It is not a universal image generator and it does not currently condition on arbitrary input images at generation time. The model learns a distribution of nail sequences for digit classes.

That constraint is also what makes the project clean: the generated output is a real continuous path, not just pixels pretending to be string art.

## Roadmap

- Condition generation on an input image embedding, not only a digit label.
- Train on EMNIST letters and custom symbols.
- Add SVG export for plotters and physical string-art machines.
- Add beam search with geometric penalties.
- Add a raster critic or reconstruction-aware loss.
- Build a small gallery script for generating many seeded variants.

## Star The Project

If you like compact generative models, algorithmic art, or physical-looking AI outputs, give Nitograph a star. It helps the project reach people who enjoy the same mix of geometry, deep learning, and visual craft.
