"""Hugging Face tokenizer adapter with exact offsets."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from sah_rag.chunking.tokenization import TokenSpan


class HuggingFaceOffsetTokenizer:
    """Adapt a fast Hugging Face tokenizer to OffsetTokenizer."""

    def __init__(
        self,
        tokenizer: Any,
        *,
        model_id: str,
        revision: str | None = None,
    ) -> None:
        if not getattr(tokenizer, "is_fast", False):
            raise ValueError(
                "Offset tokenization requires a fast tokenizer."
            )

        if not model_id.strip():
            raise ValueError(
                "model_id cannot be empty."
            )

        self._tokenizer = tokenizer
        self._model_id = model_id
        self._revision = revision

    @property
    def tokenizer_id(self) -> str:
        """Return a stable tokenizer identity."""

        revision = (
            self._revision
            if self._revision is not None
            else "default"
        )

        return (
            f"huggingface:{self._model_id}"
            f"@{revision}:fast"
        )

    @property
    def special_tokens_count(self) -> int:
        """Return special tokens added to one encoded sequence."""

        return int(
            self._tokenizer.num_special_tokens_to_add(
                pair=False
            )
        )

    @property
    def model_max_length(self) -> int:
        """Return the configured maximum sequence length."""

        return int(self._tokenizer.model_max_length)

    def tokenize_with_offsets(
        self,
        text: str,
    ) -> tuple[TokenSpan, ...]:
        """Tokenize content without adding model special tokens."""

        encoded = self._tokenizer(
            text,
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
            return_offsets_mapping=True,
            truncation=False,
        )

        input_ids = encoded["input_ids"]
        offsets = encoded["offset_mapping"]

        if not isinstance(input_ids, Sequence):
            raise TypeError(
                "Tokenizer input_ids must be a sequence."
            )

        if not isinstance(offsets, Sequence):
            raise TypeError(
                "Tokenizer offsets must be a sequence."
            )

        if len(input_ids) != len(offsets):
            raise ValueError(
                "Tokenizer IDs and offsets have "
                "different lengths."
            )

        tokens: list[TokenSpan] = []

        for _input_id, offset in zip(
            input_ids,
            offsets,
            strict=True,
        ):
            if (
                not isinstance(offset, Sequence)
                or len(offset) != 2
            ):
                raise TypeError(
                    "Every offset must contain start and end."
                )

            start = int(offset[0])
            end = int(offset[1])

            if end <= start:
                continue

            token_text = text[start:end]

            if not token_text:
                continue

            tokens.append(
                TokenSpan(
                    token_index=len(tokens),
                    text=token_text,
                    start=start,
                    end=end,
                )
            )

        return tuple(tokens)
