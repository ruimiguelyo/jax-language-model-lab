"""Small utilities shared by training and experiment scripts."""

from __future__ import annotations

import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import jax
import numpy as np
import psutil


def set_seed(seed: int) -> tuple[np.random.Generator, jax.Array]:
    return np.random.default_rng(seed), jax.random.PRNGKey(seed)


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value)!r}")


def write_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, default=json_default)
        handle.write("\n")


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def environment_info() -> dict[str, Any]:
    try:
        git_commit = (
            subprocess.run(
                ["git", "rev-parse", "HEAD"],
                check=False,
                capture_output=True,
                text=True,
            ).stdout.strip()
            or None
        )
    except OSError:
        git_commit = None

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "jax_version": jax.__version__,
        "jax_backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "cpu_count": psutil.cpu_count(logical=True),
        "memory_gb": round(psutil.virtual_memory().total / 2**30, 2),
        "git_commit": git_commit,
    }
