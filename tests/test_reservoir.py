from __future__ import annotations

import numpy as np

from jlm.continual.reservoir import ReplayExample, ReservoirBuffer
from jlm.continual.strategies import reservoir_replay_blocks
from jlm.data import TokenBlocks


def make_blocks(count: int = 20, seq_len: int = 4) -> TokenBlocks:
    inputs = np.arange(count * seq_len, dtype=np.int32).reshape(count, seq_len)
    labels = (inputs + 1) % 32
    return TokenBlocks(inputs, labels, np.ones_like(inputs, dtype=np.float32))


def test_reservoir_capacity_and_stream_count() -> None:
    buffer = ReservoirBuffer(capacity=5, seed=0)
    for index in range(20):
        value = np.full((4,), index, dtype=np.int32)
        buffer.add(ReplayExample(value, value + 1, np.ones(4, dtype=np.float32)))
    assert len(buffer) == 5
    assert buffer.seen == 20
    sample = buffer.sample(7, seed=1)
    assert sample.input_ids.shape == (7, 4)


def test_replay_strategy_preserves_fixed_example_budget() -> None:
    general = make_blocks()
    target = make_blocks()
    buffer = ReservoirBuffer(capacity=8, seed=1)
    buffer.add_blocks(general)
    mixed = reservoir_replay_blocks(target, buffer, total_examples=10, replay_ratio=0.2, seed=2)
    assert mixed.num_examples == 10
    assert mixed.seq_len == 4
