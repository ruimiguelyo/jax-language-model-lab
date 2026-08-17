"""Integration with a fixed Hugging Face tokenizer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class FixedTokenizer:
    tokenizer: Any

    @classmethod
    def from_pretrained(cls, name: str = "gpt2", cache_dir: str | None = None) -> FixedTokenizer:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(name, cache_dir=cache_dir, use_fast=True)
        if tokenizer.eos_token_id is None:
            raise ValueError(f"Tokenizer {name!r} does not define an EOS token")
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        return cls(tokenizer)

    @property
    def vocab_size(self) -> int:
        return int(len(self.tokenizer))

    @property
    def eos_token_id(self) -> int:
        return int(self.tokenizer.eos_token_id)

    @property
    def pad_token_id(self) -> int:
        return int(self.tokenizer.pad_token_id)

    def encode(self, text: str) -> list[int]:
        return list(self.tokenizer.encode(text, add_special_tokens=False))

    def decode(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=False)
