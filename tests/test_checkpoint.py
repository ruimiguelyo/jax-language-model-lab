from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from jlm.checkpoint import restore_checkpoint, save_checkpoint
from jlm.config import ModelConfig, TrainConfig
from jlm.training import create_train_state, train_step


def test_checkpoint_save_load_and_resume(tmp_path) -> None:
    model_config = ModelConfig(
        vocab_size=12, max_seq_len=4, d_model=16, n_heads=4, n_layers=1, d_ff=32
    )
    train_config = TrainConfig(batch_size=2, total_steps=10, warmup_steps=0, weight_decay=0.0)
    _, state = create_train_state(model_config, train_config, seed=5)
    batch = {
        "input_ids": jnp.asarray([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=jnp.int32),
        "labels": jnp.asarray([[2, 3, 4, 5], [3, 2, 1, 0]], dtype=jnp.int32),
        "loss_mask": jnp.ones((2, 4), dtype=jnp.float32),
    }
    state, _ = train_step(state, batch)
    metadata = {"seed": 5, "purpose": "checkpoint-test"}
    save_checkpoint(state, tmp_path, metadata=metadata)

    _, fresh_state = create_train_state(model_config, train_config, seed=99)
    restored, restored_metadata = restore_checkpoint(fresh_state, tmp_path)
    assert int(restored.step) == int(state.step)
    assert restored_metadata == metadata
    assert all(
        np.allclose(np.asarray(expected), np.asarray(actual))
        for expected, actual in zip(
            jax.tree_util.tree_leaves(state.params),
            jax.tree_util.tree_leaves(restored.params),
        )
    )

    resumed, _ = train_step(restored, batch)
    assert int(resumed.step) == int(state.step) + 1
