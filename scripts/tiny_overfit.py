from __future__ import annotations

import argparse
from pathlib import Path

import jax
import numpy as np

from jlm.config import ExperimentConfig, ModelConfig, TrainConfig, save_config
from jlm.data import TokenBlocks, cycling_batches
from jlm.metrics import append_jsonl
from jlm.training import create_train_state, eval_step, train_step
from jlm.utils import environment_info, write_json


def build_tiny_blocks(config: ModelConfig) -> TokenBlocks:
    inputs = np.asarray([[0, 1, 2, 3] * (config.max_seq_len // 4)] * 4, dtype=np.int32)
    labels = np.roll(inputs, -1, axis=1)
    labels[:, -1] = 4
    return TokenBlocks(
        input_ids=inputs,
        labels=labels,
        loss_mask=np.ones_like(labels, dtype=np.float32),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Overfit a tiny synthetic language-model dataset.")
    parser.add_argument("--output", default="results/tiny_overfit")
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_config = ModelConfig(
        vocab_size=8,
        max_seq_len=16,
        d_model=32,
        n_heads=4,
        n_layers=2,
        d_ff=64,
    )
    train_config = TrainConfig(
        seed=7,
        batch_size=4,
        learning_rate=0.01,
        weight_decay=0.0,
        warmup_steps=0,
        total_steps=args.steps,
    )
    blocks = build_tiny_blocks(model_config)
    _, state = create_train_state(model_config, train_config, train_config.seed)
    evaluation_batch = next(iter(cycling_batches(blocks, 4, seed=0, shuffle=False)))
    initial = eval_step(
        state, {key: jax.numpy.asarray(value) for key, value in evaluation_batch.items()}
    )

    batches = cycling_batches(blocks, train_config.batch_size, train_config.seed, shuffle=False)
    for _ in range(args.steps):
        state, metrics = train_step(
            state,
            {key: jax.numpy.asarray(value) for key, value in next(batches).items()},
        )
        jax.block_until_ready(metrics["loss"])

    final = eval_step(
        state, {key: jax.numpy.asarray(value) for key, value in evaluation_batch.items()}
    )
    result = {
        "initial_loss": float(initial["loss"]),
        "final_loss": float(final["loss"]),
        "initial_perplexity": float(initial["perplexity"]),
        "final_perplexity": float(final["perplexity"]),
        "steps": args.steps,
        "seed": train_config.seed,
    }
    save_config(
        ExperimentConfig(model=model_config, train=train_config), output_dir / "config.yaml"
    )
    write_json(output_dir / "environment.json", environment_info())
    write_json(output_dir / "summary.json", result)
    append_jsonl(output_dir / "metrics.jsonl", result)
    print(result)

    if result["final_loss"] >= result["initial_loss"]:
        raise SystemExit("Tiny overfit did not reduce the loss")


if __name__ == "__main__":
    main()
