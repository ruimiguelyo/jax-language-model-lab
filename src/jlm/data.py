"""Text loading, token packing and deterministic batch iteration."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class TokenBlocks:
    input_ids: np.ndarray
    labels: np.ndarray
    loss_mask: np.ndarray

    @property
    def num_examples(self) -> int:
        return int(self.input_ids.shape[0])

    @property
    def seq_len(self) -> int:
        return int(self.input_ids.shape[1])

    @property
    def token_count(self) -> int:
        return int(self.loss_mask.sum())

    def complete(self) -> TokenBlocks:
        """Return only blocks whose targets contain no padding."""

        complete_mask = self.loss_mask.sum(axis=1) == self.seq_len
        if not np.any(complete_mask):
            raise ValueError("No complete blocks are available")
        return TokenBlocks(
            input_ids=self.input_ids[complete_mask],
            labels=self.labels[complete_mask],
            loss_mask=self.loss_mask[complete_mask],
        )


def _normalise_texts(texts: Iterable[str]) -> list[str]:
    return [text.strip() for text in texts if text and text.strip()]


def documents_to_blocks(
    documents: Iterable[str],
    tokenizer,
    seq_len: int,
) -> TokenBlocks:
    if seq_len < 1:
        raise ValueError("seq_len must be positive")

    token_stream: list[int] = []
    token_mask: list[int] = []
    for document in documents:
        ids = tokenizer.encode(document)
        if not ids:
            continue
        token_stream.extend(ids)
        token_mask.extend([1] * len(ids))
        token_stream.append(tokenizer.eos_token_id)
        token_mask.append(1)

    if not token_stream:
        raise ValueError("No tokens were produced from the documents")

    block_length = seq_len + 1
    num_blocks = (len(token_stream) + block_length - 1) // block_length
    padded_length = num_blocks * block_length
    pad_length = padded_length - len(token_stream)
    token_stream.extend([tokenizer.pad_token_id] * pad_length)
    token_mask.extend([0] * pad_length)

    tokens = np.asarray(token_stream, dtype=np.int32).reshape(num_blocks, block_length)
    masks = np.asarray(token_mask, dtype=np.float32).reshape(num_blocks, block_length)
    return TokenBlocks(
        input_ids=tokens[:, :-1],
        labels=tokens[:, 1:],
        loss_mask=masks[:, 1:],
    )


def iter_batches(
    blocks: TokenBlocks,
    batch_size: int,
    seed: int,
    shuffle: bool = True,
    drop_remainder: bool = True,
) -> Iterator[dict[str, np.ndarray]]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    indices = np.arange(blocks.num_examples)
    if shuffle:
        np.random.default_rng(seed).shuffle(indices)

    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start : start + batch_size]
        if len(batch_indices) < batch_size and drop_remainder:
            continue
        yield {
            "input_ids": blocks.input_ids[batch_indices],
            "labels": blocks.labels[batch_indices],
            "loss_mask": blocks.loss_mask[batch_indices],
        }


def cycling_batches(
    blocks: TokenBlocks,
    batch_size: int,
    seed: int,
    shuffle: bool = True,
) -> Iterator[dict[str, np.ndarray]]:
    epoch = 0
    while True:
        yielded = False
        for batch in iter_batches(blocks, batch_size, seed + epoch, shuffle=shuffle):
            yielded = True
            yield batch
        if not yielded:
            raise ValueError("The dataset must contain at least one complete batch")
        epoch += 1


def read_jsonl_documents(path: str | Path) -> list[str]:
    documents: list[str] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            text = record.get("text", "")
            if text and text.strip():
                documents.append(text.strip())
    return documents


def write_jsonl_documents(path: str | Path, documents: Iterable[str]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for text in _normalise_texts(documents):
            handle.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
