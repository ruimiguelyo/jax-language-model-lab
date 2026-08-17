"""Configuration loading and serialization."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    vocab_size: int = 50_257
    max_seq_len: int = 64
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 256
    precision: str = "float32"
    tie_embeddings: bool = True


@dataclass
class TrainConfig:
    seed: int = 42
    batch_size: int = 4
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    grad_clip_norm: float = 1.0
    warmup_steps: int = 10
    total_steps: int = 100
    eval_every: int = 25
    checkpoint_every: int = 25
    log_every: int = 10
    precision: str = "float32"


@dataclass
class DataConfig:
    tokenizer_name: str = "gpt2"
    general_dataset: str = "wikitext"
    general_config: str = "wikitext-2-raw-v1"
    target_dataset: str = "ag_news"
    target_config: str | None = None
    seq_len: int = 64
    max_train_documents: int | None = None
    max_validation_documents: int | None = None
    cache_dir: str = "data/raw"


@dataclass
class ContinualConfig:
    adaptation_tokens: int = 20_000
    replay_ratio: float = 0.2
    reservoir_capacity: int = 256
    eval_fractions: list[float] = field(default_factory=lambda: [0.0, 0.25, 0.5, 0.75, 1.0])
    joint_target_ratio: float = 0.8


@dataclass
class ExperimentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    data: DataConfig = field(default_factory=DataConfig)
    continual: ContinualConfig = field(default_factory=ContinualConfig)


def _update_dataclass(instance: Any, values: dict[str, Any]) -> Any:
    for key, value in values.items():
        if not hasattr(instance, key):
            raise KeyError(f"Unknown configuration key: {key}")
        current = getattr(instance, key)
        if hasattr(current, "__dataclass_fields__"):
            _update_dataclass(current, value)
        else:
            setattr(instance, key, value)
    return instance


def load_config(path: str | Path | None = None) -> ExperimentConfig:
    config = ExperimentConfig()
    if path is None:
        return config
    with Path(path).open("r", encoding="utf-8") as handle:
        values = yaml.safe_load(handle) or {}
    return _update_dataclass(config, values)


def config_dict(config: ExperimentConfig) -> dict[str, Any]:
    return asdict(config)


def save_config(config: ExperimentConfig, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config_dict(config), handle, sort_keys=False)
