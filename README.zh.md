# Nitograph

**Nitograph 将 MNIST 数字转换成霓虹 string-art 图像，并训练一个小型 class-conditioned Transformer 来生成新的连续绕线序列。**

[English](README.md) | [Русский](README.ru.md)

![数据集预览](images/dataset_preview_img.png)

![生成的数字 7](images/digit_7_4k.png)

## 为什么有意思

Nitograph 不是普通的 text-to-image 玩具。它是一个面向特定物理表达的紧凑生成式 pipeline：一根线围绕一圈钉子连续移动。

- 将 MNIST 数字转换为有序的钉子到钉子路径。
- 在 tokenized string-art 序列上训练 autoregressive Transformer。
- 只根据数字类别生成新的路径。
- 使用 batched Matplotlib line layers 渲染清晰的 4096x4096 霓虹图像。
- 将完整钉子序列保存为 JSON，方便复现和迁移。

一句话：**image -> string-art encoder -> sequence dataset -> causal Transformer -> generated thread path -> 4K render**。

## 演示资源

预览图片位于 [`images/`](images):

- [`images/dataset_preview_img.png`](images/dataset_preview_img.png) 展示 encoded dataset 示例。
- [`images/digit_7_4k.png`](images/digit_7_4k.png) 展示生成的 4K 输出图像。

模型权重和真实输出示例可以在 [`model_and_output_examples/`](model_and_output_examples) 中找到：

- [`model_and_output_examples/string_ai.pt`](model_and_output_examples/string_ai.pt) - 训练好的 checkpoint。
- [`model_and_output_examples/digit_7_4k.png`](model_and_output_examples/digit_7_4k.png) - 生成的 render。
- [`model_and_output_examples/digit_7_4k.json`](model_and_output_examples/digit_7_4k.json) - 精确的 nail sequence。

## 安装

推荐使用 Python 3.11 或 3.12。

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/macOS
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

如果使用 NVIDIA GPU，建议先从 PyTorch 官方安装器安装与你的 CUDA 版本匹配的 PyTorch，然后再安装其他依赖。

## 快速开始

### 1. 生成数据集

快速 smoke run：

```bash
python prepare_dataset.py --samples 100 --max-lines 120 --output smoke_dataset.npz
python preview_dataset.py --dataset smoke_dataset.npz --output dataset_preview.png
```

更适合训练的版本：

```bash
python prepare_dataset.py --samples 5000 --max-lines 240
python preview_dataset.py --dataset string_dataset.npz --output dataset_preview.png
```

默认情况下，数据集构建器会为每个数字创建 canonical prototypes。这适合当前模型，因为模型按类别标签进行 conditioning，而不是按完整输入图像生成。

### 2. 训练 Transformer

```bash
python train_ai.py --epochs 15 --batch-size 32
```

低显存 GPU：

```bash
python train_ai.py --epochs 15 --batch-size 8
```

适合 CPU 的较小模型：

```bash
python train_ai.py --epochs 10 --batch-size 16 --embed-dim 128 --heads 4 --layers 3 --ff-dim 384
```

训练会写出：

- `string_ai.pt` - validation loss 最好的 checkpoint。
- `string_ai_last.pt` - 最新 checkpoint。

### 3. 生成 4K 数字

```bash
python generate.py 7
```

会生成：

- `digit_7_4k.png` - 4096x4096 渲染图。
- `digit_7_4k.json` - 精确的钉子序列。

Sampling 示例：

```bash
python generate.py 3 --temperature 0.70 --top-k 16 --seed 10
python generate.py 9 --temperature 1.00 --top-k 48 --seed 77
python generate.py 4 --temperature 0 --top-k 1
```

`--temperature 0` 会启用 deterministic greedy decoding。它可复现，但 sampling 通常能生成更丰富的线条路径。

## 工作原理

### String-Art Encoder

`prepare_dataset.py` 下载 MNIST，将钉子放在圆周上，预计算所有钉子对之间的 thick Bresenham paths，然后 greedily 选择最能改善目标数字重建效果的下一条弦。

Encoder 包含一些实用约束：

- 最小 circular nail gap；
- 禁止立即 `A -> B -> A` 回退；
- 抑制最近使用过的边；
- length-normalized scoring；
- 对过度绘制像素的 false-positive penalty；
- 压缩 `.npz` 数据集输出。

### Sequence Model

`train_ai.py` 训练一个 compact causal Transformer：

- 数字 `0..9` 使用独立 class tokens；
- nail tokens `0..255`；
- 显式 `EOS` 和 `PAD` tokens；
- validation split；
- 可用时启用 CUDA AMP；
- gradient clipping；
- 保存 best checkpoint。

### Renderer

`generate.py` 使用 temperature、top-k 和 repetition penalty 采样 nail sequence，然后渲染 4K 霓虹 string-art 图像。Renderer 使用 batched `LineCollection` layers 制造 glow，并将路径导出为 JSON。

## 项目结构

```text
nitograph/
├── geometry.py                 # 钉子几何、path tables、encoder
├── model.py                    # mini causal Transformer
├── prepare_dataset.py          # MNIST -> tokenized string-art dataset
├── preview_dataset.py          # 数据集可视化预览
├── train_ai.py                 # training loop
├── generate.py                 # sequence sampling 和 4K rendering
├── images/                     # README 预览图片
├── model_and_output_examples/  # checkpoint 和真实输出示例
├── requirements.txt
├── README.md
├── README.ru.md
└── README.zh.md
```

## 可复现性

- Dataset generation 和 sampling 都支持 `--seed`。
- 生成的 JSON 文件保存 digit、nail count、line count、sampling settings、seed 和完整钉子列表。
- 渲染尺寸由 `--size-inches` 和 `--dpi` 控制；默认 `8 * 512` 输出 4096x4096 pixels。

## 真实范围

Nitograph 是一个聚焦的 research/art MVP。它不是通用图像生成器，目前也不会根据任意输入图片进行生成。模型学习的是数字类别对应的 nail sequences 分布。

这个限制也让项目更干净：输出是真正连续的线条路径，而不是看起来像 string art 的普通像素图。

## Roadmap

- 使用输入图像 embedding 进行 conditioning，而不只是 digit label。
- 在 EMNIST letters 和 custom symbols 上训练。
- 添加 SVG export，用于 plotters 和物理 string-art 机器。
- 加入带 geometric penalties 的 beam search。
- 添加 raster critic 或 reconstruction-aware loss。
- 构建 gallery script，用不同 seed 生成大量 variants。

## Star The Project

如果你喜欢 compact generative models、algorithmic art，或者看起来可以真实制作的 AI outputs，请给 Nitograph 一个 star。它能帮助项目被更多喜欢 geometry、deep learning 和 visual craft 的人看到。
