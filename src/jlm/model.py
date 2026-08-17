"""A small decoder-only Transformer implemented with Flax and JAX."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from flax import linen as nn

from jlm.config import ModelConfig


def dtype_from_name(name: str) -> Any:
    normalized = name.lower()
    if normalized in {"float32", "fp32"}:
        return jnp.float32
    if normalized in {"bfloat16", "bf16"}:
        return jnp.bfloat16
    if normalized in {"float16", "fp16"}:
        return jnp.float16
    raise ValueError(f"Unsupported precision: {name}")


class CausalSelfAttention(nn.Module):
    d_model: int
    n_heads: int
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        batch_size, seq_len, _ = x.shape
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        head_dim = self.d_model // self.n_heads

        qkv = nn.Dense(
            features=3 * self.d_model,
            use_bias=False,
            dtype=self.dtype,
            param_dtype=jnp.float32,
            name="qkv",
        )(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)

        def split_heads(value: jax.Array) -> jax.Array:
            return value.reshape(batch_size, seq_len, self.n_heads, head_dim).transpose(0, 2, 1, 3)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)
        scores = jnp.einsum("bhqd,bhkd->bhqk", q, k)
        scores = scores / jnp.sqrt(jnp.asarray(head_dim, dtype=self.dtype))

        causal_mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=bool))
        scores = jnp.where(causal_mask[None, None, :, :], scores, jnp.finfo(scores.dtype).min)
        attention = jax.nn.softmax(scores, axis=-1)
        output = jnp.einsum("bhqk,bhkd->bhqd", attention, v)
        output = output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.d_model)

        return nn.Dense(
            features=self.d_model,
            use_bias=False,
            dtype=self.dtype,
            param_dtype=jnp.float32,
            name="out",
        )(output)


class TransformerBlock(nn.Module):
    config: ModelConfig
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        attention_input = nn.LayerNorm(
            dtype=self.dtype, param_dtype=jnp.float32, name="attention_norm"
        )(x)
        x = x + CausalSelfAttention(
            d_model=self.config.d_model,
            n_heads=self.config.n_heads,
            dtype=self.dtype,
            name="attention",
        )(attention_input)

        mlp_input = nn.LayerNorm(dtype=self.dtype, param_dtype=jnp.float32, name="mlp_norm")(x)
        mlp = nn.Dense(
            self.config.d_ff,
            dtype=self.dtype,
            param_dtype=jnp.float32,
            name="mlp_in",
        )(mlp_input)
        mlp = jax.nn.gelu(mlp)
        mlp = nn.Dense(
            self.config.d_model,
            dtype=self.dtype,
            param_dtype=jnp.float32,
            name="mlp_out",
        )(mlp)
        return x + mlp


class DecoderOnlyTransformer(nn.Module):
    config: ModelConfig

    @nn.compact
    def __call__(self, input_ids: jax.Array, deterministic: bool = True) -> jax.Array:
        del deterministic
        _, seq_len = input_ids.shape
        if seq_len > self.config.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_seq_len={self.config.max_seq_len}"
            )

        dtype = dtype_from_name(self.config.precision)
        embedding = self.param(
            "token_embedding",
            nn.initializers.normal(stddev=0.02),
            (self.config.vocab_size, self.config.d_model),
        )
        position_embedding = self.param(
            "position_embedding",
            nn.initializers.normal(stddev=0.02),
            (self.config.max_seq_len, self.config.d_model),
        )

        x = embedding[input_ids].astype(dtype)
        x = x + position_embedding[:seq_len].astype(dtype)[None, :, :]
        for layer_index in range(self.config.n_layers):
            x = TransformerBlock(self.config, dtype=dtype, name=f"block_{layer_index}")(x)
        x = nn.LayerNorm(dtype=dtype, param_dtype=jnp.float32, name="final_norm")(x)

        if self.config.tie_embeddings:
            logits = jnp.einsum("bld,vd->blv", x, embedding.astype(dtype))
        else:
            logits = nn.Dense(
                self.config.vocab_size,
                dtype=dtype,
                param_dtype=jnp.float32,
                name="lm_head",
            )(x)
        return logits.astype(jnp.float32)


def count_parameters(params: Any) -> int:
    return int(sum(x.size for x in jax.tree_util.tree_leaves(params)))
