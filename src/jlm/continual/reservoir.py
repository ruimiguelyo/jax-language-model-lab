"""Online reservoir sampling for fixed-shape language-model examples."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from jlm.data import TokenBlocks


@dataclass
class ReplayExample:
    input_ids: np.ndarray
    labels: np.ndarray
    loss_mask: np.ndarray


class ReservoirBuffer:
    """Keep a uniform sample of a stream using Algorithm R."""

    def __init__(self, capacity: int, seed: int = 0):
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self.capacity = int(capacity)
        self.rng = np.random.default_rng(seed)
        self._items: list[ReplayExample] = []
        self.seen = 0

    def add(self, example: ReplayExample) -> None:
        self.seen += 1
        copied = ReplayExample(
            input_ids=np.asarray(example.input_ids, dtype=np.int32).copy(),
            labels=np.asarray(example.labels, dtype=np.int32).copy(),
            loss_mask=np.asarray(example.loss_mask, dtype=np.float32).copy(),
        )
        if len(self._items) < self.capacity:
            self._items.append(copied)
            return

        replacement_index = int(self.rng.integers(0, self.seen))
        if replacement_index < self.capacity:
            self._items[replacement_index] = copied

    def add_blocks(self, blocks: TokenBlocks) -> None:
        for index in range(blocks.num_examples):
            self.add(
                ReplayExample(
                    input_ids=blocks.input_ids[index],
                    labels=blocks.labels[index],
                    loss_mask=blocks.loss_mask[index],
                )
            )

    def sample(self, count: int, seed: int | None = None) -> TokenBlocks:
        if not self._items:
            raise ValueError("Cannot sample from an empty reservoir")
        if count < 1:
            raise ValueError("count must be positive")
        rng = np.random.default_rng(seed) if seed is not None else self.rng
        indices = rng.integers(0, len(self._items), size=count)
        selected = [self._items[int(index)] for index in indices]
        return TokenBlocks(
            input_ids=np.stack([item.input_ids for item in selected]),
            labels=np.stack([item.labels for item in selected]),
            loss_mask=np.stack([item.loss_mask for item in selected]),
        )

    def as_blocks(self) -> TokenBlocks:
        if not self._items:
            raise ValueError("Cannot convert an empty reservoir to blocks")
        return TokenBlocks(
            input_ids=np.stack([item.input_ids for item in self._items]),
            labels=np.stack([item.labels for item in self._items]),
            loss_mask=np.stack([item.loss_mask for item in self._items]),
        )

    def __len__(self) -> int:
        return len(self._items)


def build_reservoir(blocks: TokenBlocks, capacity: int, seed: int = 0) -> ReservoirBuffer:
    buffer = ReservoirBuffer(capacity=capacity, seed=seed)
    buffer.add_blocks(blocks)
    return buffer
