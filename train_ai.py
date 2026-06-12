from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

from model import MiniCausalTransformer, ModelConfig


class StringArtDataset(Dataset):
    def __init__(self, path: Path) -> None:
        archive = np.load(path, allow_pickle=False)
        self.tokens = torch.from_numpy(archive["tokens"].astype(np.int64))
        self.labels = torch.from_numpy(archive["labels"].astype(np.int64))
        self.metadata = json.loads(str(archive["metadata"].item()))

        if self.tokens.ndim != 2:
            raise ValueError("tokens must have shape [samples, max_sequence]")
        if len(self.tokens) != len(self.labels):
            raise ValueError("tokens and labels have different sample counts")

        self.num_nails = int(self.metadata["num_nails"])
        self.pad_token = int(self.metadata["pad_token"])
        self.class_token_offset = int(self.metadata["class_token_offset"])

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sequence = self.tokens[index]
        label = int(self.labels[index])
        class_token = self.class_token_offset + label

        # Example:
        # input  = [CLASS_7, nail_0, nail_31, ..., EOS/PAD excluded at end]
        # target = [nail_0, nail_31, ..., EOS, PAD...]
        x = torch.empty_like(sequence)
        x[0] = class_token
        x[1:] = sequence[:-1]
        y = sequence
        return x, y


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the string-art transformer.")
    parser.add_argument("--dataset", type=Path, default=Path("string_dataset.npz"))
    parser.add_argument("--output", type=Path, default=Path("string_ai.pt"))
    parser.add_argument("--last-output", type=Path, default=Path("string_ai_last.pt"))
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument("--embed-dim", type=int, default=192)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--ff-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def loss_and_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pad_token: int,
) -> tuple[torch.Tensor, float]:
    loss = F.cross_entropy(
        logits.transpose(1, 2),
        targets,
        ignore_index=pad_token,
    )

    with torch.no_grad():
        valid = targets.ne(pad_token)
        predicted = logits.argmax(dim=-1)
        correct = predicted.eq(targets) & valid
        accuracy = (
            float(correct.sum().item() / valid.sum().item())
            if valid.any()
            else 0.0
        )
    return loss, accuracy


@torch.no_grad()
def evaluate(
    model: MiniCausalTransformer,
    loader: DataLoader,
    device: torch.device,
    pad_token: int,
) -> tuple[float, float]:
    model.eval()
    losses: list[float] = []
    weighted_correct = 0.0
    weighted_tokens = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss, accuracy = loss_and_accuracy(logits, y, pad_token)

        token_count = int(y.ne(pad_token).sum().item())
        losses.append(float(loss.item()))
        weighted_correct += accuracy * token_count
        weighted_tokens += token_count

    mean_loss = float(np.mean(losses)) if losses else math.nan
    mean_accuracy = weighted_correct / max(weighted_tokens, 1)
    return mean_loss, mean_accuracy


def checkpoint_payload(
    model: MiniCausalTransformer,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    validation_loss: float,
    dataset: StringArtDataset,
    args: argparse.Namespace,
) -> dict:
    return {
        "format_version": 2,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "model_config": model.config.to_dict(),
        "data_metadata": dataset.metadata,
        "training": {
            "epoch": epoch,
            "validation_loss": validation_loss,
            "seed": args.seed,
        },
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = choose_device(args.device)

    dataset = StringArtDataset(args.dataset)
    validation_size = max(1, round(len(dataset) * args.validation_fraction))
    if validation_size >= len(dataset):
        raise ValueError("Dataset is too small for the requested validation split")
    training_size = len(dataset) - validation_size

    generator = torch.Generator().manual_seed(args.seed)
    train_set, validation_set = random_split(
        dataset,
        [training_size, validation_size],
        generator=generator,
    )

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=pin_memory,
        persistent_workers=args.workers > 0,
    )
    validation_loader = DataLoader(
        validation_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=pin_memory,
        persistent_workers=args.workers > 0,
    )

    config = ModelConfig(
        num_nails=dataset.num_nails,
        max_seq_len=dataset.tokens.shape[1],
        num_classes=int(dataset.metadata.get("num_classes", 10)),
        embed_dim=args.embed_dim,
        num_heads=args.heads,
        num_layers=args.layers,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
    )
    model = MiniCausalTransformer(config).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, args.epochs),
        eta_min=args.learning_rate * 0.10,
    )

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    print(f"Device: {device}")
    print(f"Training samples: {training_size}")
    print(f"Validation samples: {validation_size}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    best_validation_loss = float("inf")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.last_output.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_accuracy = 0.0
        batch_count = 0

        progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{args.epochs}",
            leave=False,
        )
        for x, y in progress:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=use_amp,
            ):
                logits = model(x)
                loss, accuracy = loss_and_accuracy(
                    logits,
                    y,
                    dataset.pad_token,
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            running_loss += float(loss.item())
            running_accuracy += accuracy
            batch_count += 1
            progress.set_postfix(
                loss=f"{running_loss / batch_count:.4f}",
                acc=f"{running_accuracy / batch_count:.1%}",
            )

        validation_loss, validation_accuracy = evaluate(
            model,
            validation_loader,
            device,
            dataset.pad_token,
        )
        scheduler.step()

        train_loss = running_loss / max(batch_count, 1)
        train_accuracy = running_accuracy / max(batch_count, 1)
        print(
            f"Epoch {epoch:02d} | "
            f"train loss {train_loss:.4f} | "
            f"train acc {train_accuracy:.1%} | "
            f"val loss {validation_loss:.4f} | "
            f"val acc {validation_accuracy:.1%}"
        )

        payload = checkpoint_payload(
            model,
            optimizer,
            epoch,
            validation_loss,
            dataset,
            args,
        )
        torch.save(payload, args.last_output)

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            torch.save(payload, args.output)
            print(f"  New best checkpoint: {args.output}")

    print(f"\nBest validation loss: {best_validation_loss:.4f}")
    print(f"Best model: {args.output}")


if __name__ == "__main__":
    main()
