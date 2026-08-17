"""Checkpoint persistence with a small JSON metadata sidecar."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from flax.training import checkpoints

from jlm.training import TrainState
from jlm.utils import read_json, write_json


def save_checkpoint(
    state: TrainState,
    directory: str | Path,
    metadata: dict[str, Any] | None = None,
    keep: int = 3,
) -> Path:
    destination = Path(directory).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    step = int(state.step)
    checkpoint_path = Path(
        checkpoints.save_checkpoint(
            ckpt_dir=str(destination),
            target=state,
            step=step,
            keep=keep,
            overwrite=True,
        )
    )
    if metadata is not None:
        write_json(destination / f"metadata_{step}.json", metadata)
    return checkpoint_path


def restore_checkpoint(
    state: TrainState,
    directory: str | Path,
) -> tuple[TrainState, dict[str, Any]]:
    source = Path(directory).resolve()
    latest = checkpoints.latest_checkpoint(str(source))
    if latest is None:
        raise FileNotFoundError(f"No checkpoint found in {source}")
    restored = checkpoints.restore_checkpoint(str(source), target=state)
    step = int(restored.step)
    metadata_path = source / f"metadata_{step}.json"
    metadata = read_json(metadata_path) if metadata_path.exists() else {}
    return restored, metadata
