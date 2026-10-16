from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RunaConfig:
    vocab_size: int
    max_seq_len: int
    n_layers: int
    d_model: int
    n_heads: int
    d_ff: int
    dropout: float = 0.0
    norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    tie_embeddings: bool = True

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for rotary embeddings")
        if self.max_seq_len < 1:
            raise ValueError("max_seq_len must be positive")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "RunaConfig":
        names = {field.name for field in fields(cls)}
        return cls(**{name: values[name] for name in names if name in values})

    @classmethod
    def from_json(cls, path: str | Path) -> "RunaConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            values = json.load(handle)
        return cls.from_dict(values)

