"""Construction of fixed-budget continual-pretraining streams."""

from __future__ import annotations

import numpy as np

from jlm.continual.reservoir import ReservoirBuffer
from jlm.data import TokenBlocks


def take_blocks(blocks: TokenBlocks, count: int) -> TokenBlocks:
    """Take a deterministic prefix, cycling when a small corpus is exhausted."""

    if count < 1:
        raise ValueError("count must be positive")
    indices = np.arange(count) % blocks.num_examples
    return TokenBlocks(
        input_ids=blocks.input_ids[indices],
        labels=blocks.labels[indices],
        loss_mask=blocks.loss_mask[indices],
    )


def _shuffle_blocks(blocks: TokenBlocks, seed: int) -> TokenBlocks:
    indices = np.arange(blocks.num_examples)
    np.random.default_rng(seed).shuffle(indices)
    return TokenBlocks(
        input_ids=blocks.input_ids[indices],
        labels=blocks.labels[indices],
        loss_mask=blocks.loss_mask[indices],
    )


def sequential_blocks(target_blocks: TokenBlocks, total_examples: int) -> TokenBlocks:
    return take_blocks(target_blocks, total_examples)


def reservoir_replay_blocks(
    target_blocks: TokenBlocks,
    reservoir: ReservoirBuffer,
    total_examples: int,
    replay_ratio: float,
    seed: int,
) -> TokenBlocks:
    if not 0.0 <= replay_ratio < 1.0:
        raise ValueError("replay_ratio must be in [0, 1)")
    replay_examples = int(round(total_examples * replay_ratio))
    target_examples = total_examples - replay_examples
    target = take_blocks(target_blocks, max(target_examples, 1))
    if replay_examples == 0:
        return target
    replay = reservoir.sample(replay_examples, seed=seed)
    return _shuffle_blocks(
        TokenBlocks(
            input_ids=np.concatenate([target.input_ids, replay.input_ids]),
            labels=np.concatenate([target.labels, replay.labels]),
            loss_mask=np.concatenate([target.loss_mask, replay.loss_mask]),
        ),
        seed,
    )


def joint_training_blocks(
    general_blocks: TokenBlocks,
    target_blocks: TokenBlocks,
    total_examples: int,
    target_ratio: float,
    seed: int,
) -> TokenBlocks:
    if not 0.0 < target_ratio < 1.0:
        raise ValueError("target_ratio must be in (0, 1)")
    target_examples = int(round(total_examples * target_ratio))
    general_examples = total_examples - target_examples
    target = take_blocks(target_blocks, max(target_examples, 1))
    general = take_blocks(general_blocks, max(general_examples, 1))
    return _shuffle_blocks(
        TokenBlocks(
            input_ids=np.concatenate([general.input_ids, target.input_ids]),
            labels=np.concatenate([general.labels, target.labels]),
            loss_mask=np.concatenate([general.loss_mask, target.loss_mask]),
        ),
        seed,
    )


def strategy_blocks(
    strategy: str,
    general_blocks: TokenBlocks,
    target_blocks: TokenBlocks,
    reservoir: ReservoirBuffer,
    total_examples: int,
    replay_ratio: float,
    joint_target_ratio: float,
    seed: int,
) -> TokenBlocks:
    if strategy == "sequential":
        return sequential_blocks(target_blocks, total_examples)
    if strategy == "reservoir_replay":
        return reservoir_replay_blocks(
            target_blocks,
            reservoir,
            total_examples,
            replay_ratio,
            seed,
        )
    if strategy == "joint_training_reference":
        return joint_training_blocks(
            general_blocks,
            target_blocks,
            total_examples,
            joint_target_ratio,
            seed,
        )
    raise ValueError(f"Unknown adaptation strategy: {strategy}")
