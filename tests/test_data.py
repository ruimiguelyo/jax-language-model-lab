from __future__ import annotations

import numpy as np

from jlm.data import documents_to_blocks


class FakeTokenizer:
    eos_token_id = 0
    pad_token_id = 0

    def encode(self, text: str) -> list[int]:
        return [(ord(character) % 7) + 1 for character in text]


def test_documents_are_packed_into_fixed_length_lm_examples() -> None:
    blocks = documents_to_blocks(["abcd", "efgh"], FakeTokenizer(), seq_len=4)
    assert blocks.input_ids.shape[1] == 4
    assert blocks.labels.shape == blocks.input_ids.shape
    assert blocks.loss_mask.shape == blocks.input_ids.shape
    assert blocks.token_count > 0


def test_padding_is_excluded_from_complete_blocks() -> None:
    blocks = documents_to_blocks(["abcde"], FakeTokenizer(), seq_len=4)
    complete = blocks.complete()
    assert complete.num_examples < blocks.num_examples or np.all(complete.loss_mask == 1)
    assert np.all(complete.loss_mask == 1)
