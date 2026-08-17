"""Language-modeling losses and metrics."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optax


def masked_cross_entropy(
    logits: jax.Array,
    labels: jax.Array,
    loss_mask: jax.Array | None = None,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    labels_one_hot = jax.nn.one_hot(labels, num_classes=logits.shape[-1])
    token_losses = optax.softmax_cross_entropy(logits, labels_one_hot)
    if loss_mask is None:
        loss_mask = jnp.ones_like(token_losses)
    loss_mask = loss_mask.astype(token_losses.dtype)
    token_count = jnp.maximum(loss_mask.sum(), 1.0)
    loss = (token_losses * loss_mask).sum() / token_count
    metrics = {
        "loss": loss,
        "token_count": token_count,
        "perplexity": jnp.exp(jnp.minimum(loss, 20.0)),
    }
    return loss, metrics
