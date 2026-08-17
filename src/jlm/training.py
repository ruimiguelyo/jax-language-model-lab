"""JAX training and evaluation steps."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state

from jlm.config import ModelConfig, TrainConfig
from jlm.losses import masked_cross_entropy
from jlm.model import DecoderOnlyTransformer


class TrainState(train_state.TrainState):
    """TrainState alias kept explicit for checkpoint and type readability."""


def create_train_state(
    model_config: ModelConfig,
    train_config: TrainConfig,
    seed: int,
) -> tuple[DecoderOnlyTransformer, TrainState]:
    model = DecoderOnlyTransformer(model_config)
    rng = jax.random.PRNGKey(seed)
    dummy_inputs = jnp.zeros((1, model_config.max_seq_len), dtype=jnp.int32)
    params = model.init(rng, dummy_inputs)["params"]

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=train_config.learning_rate,
        warmup_steps=train_config.warmup_steps,
        decay_steps=max(train_config.total_steps, train_config.warmup_steps + 1),
        end_value=train_config.learning_rate * 0.1,
    )
    tx = optax.chain(
        optax.clip_by_global_norm(train_config.grad_clip_norm),
        optax.adamw(schedule, weight_decay=train_config.weight_decay),
    )
    state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    return model, state


@jax.jit
def train_step(
    state: TrainState, batch: dict[str, jax.Array]
) -> tuple[TrainState, dict[str, jax.Array]]:
    def loss_fn(params: Any) -> tuple[jax.Array, dict[str, jax.Array]]:
        logits = state.apply_fn(
            {"params": params},
            batch["input_ids"],
            deterministic=True,
        )
        return masked_cross_entropy(logits, batch["labels"], batch.get("loss_mask"))

    (loss, metrics), gradients = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    metrics = dict(metrics)
    metrics["grad_norm"] = optax.tree.norm(gradients)
    metrics["loss"] = loss
    return state.apply_gradients(grads=gradients), metrics


@jax.jit
def eval_step(state: TrainState, batch: dict[str, jax.Array]) -> dict[str, jax.Array]:
    logits = state.apply_fn(
        {"params": state.params},
        batch["input_ids"],
        deterministic=True,
    )
    _, metrics = masked_cross_entropy(logits, batch["labels"], batch.get("loss_mask"))
    return metrics


def to_jax_batch(batch: dict[str, Any]) -> dict[str, jax.Array]:
    return {key: jnp.asarray(value) for key, value in batch.items()}


def _as_float(value: Any) -> float:
    return float(jax.device_get(value))


def evaluate(
    state: TrainState,
    batches: Iterator[dict[str, Any]],
) -> dict[str, float]:
    total_loss = 0.0
    total_tokens = 0.0
    batches_seen = 0
    for raw_batch in batches:
        metrics = eval_step(state, to_jax_batch(raw_batch))
        token_count = _as_float(metrics["token_count"])
        total_loss += _as_float(metrics["loss"]) * token_count
        total_tokens += token_count
        batches_seen += 1
    if total_tokens == 0:
        raise ValueError("Evaluation received no valid tokens")
    loss = total_loss / total_tokens
    return {
        "loss": loss,
        "perplexity": float(jnp.exp(jnp.minimum(loss, 20.0))),
        "token_count": total_tokens,
        "batches": batches_seen,
    }


def train_steps(
    state: TrainState,
    batches: Iterator[dict[str, Any]],
    num_steps: int,
) -> tuple[TrainState, list[dict[str, float]]]:
    if num_steps < 1:
        raise ValueError("num_steps must be positive")
    history: list[dict[str, float]] = []
    for step_index in range(num_steps):
        state, metrics = train_step(state, to_jax_batch(next(batches)))
        history.append(
            {
                "step": float(step_index + 1),
                "loss": _as_float(metrics["loss"]),
                "perplexity": _as_float(metrics["perplexity"]),
                "token_count": _as_float(metrics["token_count"]),
                "grad_norm": _as_float(metrics["grad_norm"]),
            }
        )
    return state, history
