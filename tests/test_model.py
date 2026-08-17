from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from jlm.config import ModelConfig, TrainConfig
from jlm.data import TokenBlocks, cycling_batches, iter_batches
from jlm.losses import masked_cross_entropy
from jlm.model import DecoderOnlyTransformer, count_parameters
from jlm.training import create_train_state, eval_step, train_step


def small_model_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=16,
        max_seq_len=8,
        d_model=16,
        n_heads=4,
        n_layers=1,
        d_ff=32,
    )


def make_batch(batch_size: int = 2, seq_len: int = 8) -> dict[str, np.ndarray]:
    inputs = np.tile(np.arange(seq_len, dtype=np.int32), (batch_size, 1)) % 16
    labels = np.roll(inputs, -1, axis=1)
    mask = np.ones_like(labels, dtype=np.float32)
    return {"input_ids": inputs, "labels": labels, "loss_mask": mask}


def test_output_shapes_and_parameter_count() -> None:
    config = small_model_config()
    model = DecoderOnlyTransformer(config)
    variables = model.init(jax.random.PRNGKey(0), jnp.zeros((2, 8), dtype=jnp.int32))
    logits = model.apply(variables, jnp.zeros((2, 8), dtype=jnp.int32))
    assert logits.shape == (2, 8, config.vocab_size)
    assert count_parameters(variables["params"]) > 0


def test_causal_mask_blocks_future_tokens() -> None:
    config = small_model_config()
    model = DecoderOnlyTransformer(config)
    first = jnp.asarray([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=jnp.int32)
    changed_future = jnp.asarray([[1, 2, 3, 4, 9, 10, 11, 12]], dtype=jnp.int32)
    variables = model.init(jax.random.PRNGKey(1), first)
    logits_first = model.apply(variables, first)
    logits_changed = model.apply(variables, changed_future)
    np.testing.assert_allclose(logits_first[:, :4], logits_changed[:, :4], rtol=1e-5, atol=1e-5)


def test_loss_and_gradients_are_finite() -> None:
    config = small_model_config()
    model = DecoderOnlyTransformer(config)
    batch = make_batch()
    variables = model.init(jax.random.PRNGKey(2), batch["input_ids"])

    def loss_fn(params):
        logits = model.apply({"params": params}, batch["input_ids"])
        return masked_cross_entropy(logits, batch["labels"], batch["loss_mask"])[0]

    loss, gradients = jax.value_and_grad(loss_fn)(variables["params"])
    assert np.isfinite(float(loss))
    assert all(
        np.all(np.isfinite(np.asarray(leaf))) for leaf in jax.tree_util.tree_leaves(gradients)
    )


def test_training_step_returns_finite_metrics() -> None:
    model_config = small_model_config()
    train_config = TrainConfig(
        batch_size=2,
        learning_rate=1e-3,
        weight_decay=0.0,
        warmup_steps=0,
        total_steps=10,
    )
    _, state = create_train_state(model_config, train_config, seed=3)
    state, metrics = train_step(
        state, {key: jnp.asarray(value) for key, value in make_batch().items()}
    )
    assert int(state.step) == 1
    assert np.isfinite(float(metrics["loss"]))
    assert np.isfinite(float(metrics["grad_norm"]))


def test_tiny_dataset_overfit() -> None:
    model_config = ModelConfig(
        vocab_size=8,
        max_seq_len=4,
        d_model=16,
        n_heads=4,
        n_layers=1,
        d_ff=32,
    )
    train_config = TrainConfig(
        seed=4,
        batch_size=4,
        learning_rate=0.01,
        weight_decay=0.0,
        warmup_steps=0,
        total_steps=80,
    )
    input_ids = np.asarray([[0, 1, 2, 3]] * 4, dtype=np.int32)
    labels = np.asarray([[1, 2, 3, 4]] * 4, dtype=np.int32)
    blocks = TokenBlocks(input_ids, labels, np.ones_like(labels, dtype=np.float32))
    _, state = create_train_state(model_config, train_config, seed=train_config.seed)

    initial = eval_step(
        state,
        {key: jnp.asarray(value) for key, value in next(iter_batches(blocks, 4, 0, False)).items()},
    )
    batches = cycling_batches(
        blocks, train_config.batch_size, seed=train_config.seed, shuffle=False
    )
    for _ in range(train_config.total_steps):
        state, _ = train_step(
            state, {key: jnp.asarray(value) for key, value in next(batches).items()}
        )
    final = eval_step(
        state,
        {key: jnp.asarray(value) for key, value in next(iter_batches(blocks, 4, 0, False)).items()},
    )
    assert float(final["loss"]) < float(initial["loss"]) * 0.5
