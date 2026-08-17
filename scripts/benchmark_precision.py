from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import jax
import numpy as np

from jlm.config import load_config
from jlm.metrics import append_jsonl
from jlm.training import create_train_state, eval_step, train_step
from jlm.utils import environment_info, write_json


def synthetic_batch(
    vocab_size: int, batch_size: int, seq_len: int, seed: int
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    input_ids = rng.integers(0, vocab_size, size=(batch_size, seq_len), dtype=np.int32)
    labels = rng.integers(0, vocab_size, size=(batch_size, seq_len), dtype=np.int32)
    return {
        "input_ids": input_ids,
        "labels": labels,
        "loss_mask": np.ones((batch_size, seq_len), dtype=np.float32),
    }


def device_memory_stats() -> dict[str, int] | None:
    stats = jax.devices()[0].memory_stats()
    if not stats:
        return None
    selected = {}
    for key in ("bytes_in_use", "peak_bytes_in_use", "bytes_limit"):
        if key in stats:
            selected[key] = int(stats[key])
    return selected or None


def benchmark_precision(config, precision: str, timed_steps: int, warmup_steps: int) -> dict:
    local_config = copy.deepcopy(config)
    local_config.model.precision = precision
    local_config.train.precision = precision
    _, state = create_train_state(local_config.model, local_config.train, local_config.train.seed)
    batch = synthetic_batch(
        local_config.model.vocab_size,
        local_config.train.batch_size,
        local_config.data.seq_len,
        local_config.train.seed,
    )
    batch_jax = {key: jax.numpy.asarray(value) for key, value in batch.items()}

    for _ in range(warmup_steps):
        state, metrics = train_step(state, batch_jax)
        jax.block_until_ready(metrics["loss"])

    start = time.perf_counter()
    for _ in range(timed_steps):
        state, metrics = train_step(state, batch_jax)
        jax.block_until_ready(metrics["loss"])
    elapsed = time.perf_counter() - start
    evaluation = eval_step(state, batch_jax)
    jax.block_until_ready(evaluation["loss"])
    token_count = timed_steps * local_config.train.batch_size * local_config.data.seq_len
    return {
        "precision": precision,
        "steps": timed_steps,
        "elapsed_seconds": elapsed,
        "tokens_per_second": token_count / elapsed,
        "validation_loss": float(evaluation["loss"]),
        "validation_perplexity": float(evaluation["perplexity"]),
        "memory_stats": device_memory_stats(),
        "backend": jax.default_backend(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark FP32 and BF16 training when supported.")
    parser.add_argument("--config", default="configs/real.yaml")
    parser.add_argument("--output", default="results/precision_benchmark")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--allow-cpu-bf16", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = [benchmark_precision(config, "float32", args.steps, args.warmup_steps)]
    device = jax.devices()[0]
    if device.platform != "cpu" or args.allow_cpu_bf16:
        records.append(benchmark_precision(config, "bfloat16", args.steps, args.warmup_steps))
    else:
        print("Skipping BF16: CPU BF16 was not requested explicitly.")
    for record in records:
        append_jsonl(output_dir / "metrics.jsonl", record)
        print(record)
    write_json(output_dir / "environment.json", environment_info())
    write_json(
        output_dir / "summary.json",
        {
            "records": records,
            "bf16_executed": any(record["precision"] == "bfloat16" for record in records),
            "note": "Memory statistics are reported only when the active JAX device exposes them.",
        },
    )


if __name__ == "__main__":
    main()
