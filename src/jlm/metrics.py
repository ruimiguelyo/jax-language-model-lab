"""Metric persistence, continual-learning metrics and plots."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def append_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def forgetting(baseline_loss: float, current_loss: float) -> float:
    """Return loss increase on the original domain: F_t = L_t - L_0."""

    return float(current_loss - baseline_loss)


def build_continual_record(
    strategy: str,
    adaptation_tokens: int,
    general_metrics: dict[str, float],
    target_metrics: dict[str, float],
    baseline_general_loss: float,
) -> dict[str, Any]:
    return {
        "strategy": strategy,
        "adaptation_tokens": int(adaptation_tokens),
        "general_loss": general_metrics["loss"],
        "general_perplexity": general_metrics["perplexity"],
        "target_loss": target_metrics["loss"],
        "target_perplexity": target_metrics["perplexity"],
        "forgetting": forgetting(baseline_general_loss, general_metrics["loss"]),
    }


def plot_continual_metrics(records: list[dict[str, Any]], output_dir: str | Path) -> list[Path]:
    import matplotlib.pyplot as plt

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    strategies = sorted({record["strategy"] for record in records})

    for metric, ylabel, filename in [
        ("general_perplexity", "General-domain validation perplexity", "general_perplexity.png"),
        ("target_perplexity", "Target-domain validation perplexity", "target_perplexity.png"),
    ]:
        figure, axis = plt.subplots(figsize=(7, 4.5))
        for strategy in strategies:
            selected = [record for record in records if record["strategy"] == strategy]
            selected.sort(key=lambda record: record["adaptation_tokens"])
            axis.plot(
                [record["adaptation_tokens"] for record in selected],
                [record[metric] for record in selected],
                marker="o",
                label=strategy,
            )
        axis.set_xlabel("Adaptation tokens")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
        axis.legend()
        figure.tight_layout()
        path = destination / filename
        figure.savefig(path, dpi=160)
        plt.close(figure)
        paths.append(path)
    return paths
