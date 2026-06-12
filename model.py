from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class ModelConfig:
    num_nails: int
    max_seq_len: int
    num_classes: int = 10
    embed_dim: int = 192
    num_heads: int = 6
    num_layers: int = 4
    ff_dim: int = 512
    dropout: float = 0.10

    @property
    def eos_token(self) -> int:
        return self.num_nails

    @property
    def pad_token(self) -> int:
        return self.num_nails + 1

    @property
    def class_token_offset(self) -> int:
        return self.num_nails + 2

    @property
    def input_vocab_size(self) -> int:
        # nails + EOS + PAD + ten separate class tokens
        return self.num_nails + 2 + self.num_classes

    @property
    def output_vocab_size(self) -> int:
        # The model may emit a nail or EOS, but never PAD/class tokens.
        return self.num_nails + 1

    def to_dict(self) -> dict:
        return asdict(self)


class MiniCausalTransformer(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if config.embed_dim % config.num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.config = config
        self.token_embedding = nn.Embedding(
            config.input_vocab_size,
            config.embed_dim,
            padding_idx=config.pad_token,
        )
        self.position_embedding = nn.Embedding(
            config.max_seq_len,
            config.embed_dim,
        )

        layer = nn.TransformerEncoderLayer(
            d_model=config.embed_dim,
            nhead=config.num_heads,
            dim_feedforward=config.ff_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=config.num_layers,
            enable_nested_tensor=False,
        )
        self.final_norm = nn.LayerNorm(config.embed_dim)
        self.head = nn.Linear(
            config.embed_dim,
            config.output_vocab_size,
            bias=False,
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape [batch, time]")

        batch_size, sequence_length = token_ids.shape
        if sequence_length > self.config.max_seq_len:
            raise ValueError(
                f"Sequence length {sequence_length} exceeds "
                f"max_seq_len={self.config.max_seq_len}"
            )

        positions = torch.arange(
            sequence_length,
            device=token_ids.device,
            dtype=torch.long,
        ).unsqueeze(0)

        x = (
            self.token_embedding(token_ids)
            + self.position_embedding(positions)
        )

        # True values are hidden from attention. This is an upper-triangular
        # causal mask: a token cannot see tokens to its right.
        causal_mask = torch.triu(
            torch.ones(
                sequence_length,
                sequence_length,
                device=token_ids.device,
                dtype=torch.bool,
            ),
            diagonal=1,
        )
        padding_mask = token_ids.eq(self.config.pad_token)

        x = self.transformer(
            x,
            mask=causal_mask,
            src_key_padding_mask=padding_mask,
            is_causal=True,
        )
        return self.head(self.final_norm(x))


def config_from_dict(data: dict) -> ModelConfig:
    allowed = {
        "num_nails",
        "max_seq_len",
        "num_classes",
        "embed_dim",
        "num_heads",
        "num_layers",
        "ff_dim",
        "dropout",
    }
    return ModelConfig(**{key: value for key, value in data.items() if key in allowed})
