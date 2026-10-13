"""Deterministic oversize-document windowing for Metadata-RAG.

Frozen specification:
    baseline_audit/metadata_rag_clean_room_spec.yaml
Frozen amendment 04:
    baseline_audit/metadata_rag_clean_room_spec_amendment_04.yaml

The frozen windowing policy is:

  * The maximum content-token budget per window is 8190 tokens, so
    that adding the two special tokens that BGE-M3 prepends to every
    input yields exactly the model maximum of 8192 tokens.
  * Token boundaries are assigned with the same pinned BGE-M3
    tokenizer that the frozen embedder already uses.
  * Windows are contiguous and exhaustive: the concatenation of the
    window character spans equals the source text exactly.
  * The splitter is a pure function of (text, tokenizer identity).
    Two runs on the same input produce byte-identical window spans.

The module never touches the network: the production splitter loads
the tokenizer lazily from a local snapshot path with
``local_files_only=True``. Tests inject a deterministic fake
tokenizer and never touch transformers.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, Final, Protocol, Self, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, model_validator

WINDOWING_VERSION: Final[str] = "0.1.0"

WINDOW_CONTENT_TOKENS: Final[int] = 8190
SPECIAL_TOKEN_COUNT: Final[int] = 2
MAXIMUM_MODEL_INPUT_TOKENS: Final[int] = 8192

TokenizerFactory = Callable[[], Any]


class MetadataRAGWindowingError(ValueError):
    """Raised when window splitting violates its frozen contract."""


class WindowSpan(BaseModel):
    """One deterministic window of a parent document.

    Fields:
        window_index: zero-based index of this window within the
            parent document.
        char_start: inclusive character start offset into the parent
            document's normalized text.
        char_end: exclusive character end offset into the parent
            document's normalized text.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=False,
    )

    window_index: int = Field(ge=0)
    char_start: int = Field(ge=0)
    char_end: int = Field(gt=0)

    @model_validator(mode="after")
    def _enforce_shape(self) -> Self:
        if self.char_end <= self.char_start:
            raise ValueError(
                "char_end must be strictly greater than char_start"
            )
        return self


@runtime_checkable
class WindowSplitterProtocol(Protocol):
    """Protocol implemented by every Metadata-RAG window splitter."""

    @property
    def max_content_tokens(self) -> int: ...

    def split(self, text: str) -> tuple[WindowSpan, ...]: ...


def _validate_text(text: str) -> None:
    if not isinstance(text, str):
        raise MetadataRAGWindowingError("text must be a string")
    if not text:
        raise MetadataRAGWindowingError("text must be non-empty")


class SingleWindowSplitter:
    """Trivial deterministic splitter: one window per input text.

    This splitter is valid on any corpus whose documents are known to
    fit within the frozen window budget. It is used by the synthetic
    acceptance run (all 12 fixture documents are under 607 tokens) and
    by any document of ``clerk``, whose maximum observed token count
    is 607.
    """

    def __init__(
        self, max_content_tokens: int = WINDOW_CONTENT_TOKENS
    ) -> None:
        if max_content_tokens < 1:
            raise MetadataRAGWindowingError(
                "max_content_tokens must be >= 1"
            )
        self._max = max_content_tokens

    @property
    def max_content_tokens(self) -> int:
        return self._max

    def split(self, text: str) -> tuple[WindowSpan, ...]:
        _validate_text(text)
        return (
            WindowSpan(
                window_index=0,
                char_start=0,
                char_end=len(text),
            ),
        )


class BgeM3WindowSplitter:
    """Window splitter backed by the pinned BGE-M3 tokenizer.

    The tokenizer is injected via a factory, so this class stays
    independent of transformers and of any particular model. The
    production factory loads the pinned BGE-M3 tokenizer from a local
    snapshot. Tests inject a fake tokenizer.
    """

    def __init__(
        self,
        *,
        tokenizer_factory: TokenizerFactory,
        max_content_tokens: int = WINDOW_CONTENT_TOKENS,
    ) -> None:
        if max_content_tokens < 1:
            raise MetadataRAGWindowingError(
                "max_content_tokens must be >= 1"
            )
        if not callable(tokenizer_factory):
            raise MetadataRAGWindowingError(
                "tokenizer_factory must be callable"
            )
        self._factory = tokenizer_factory
        self._max = max_content_tokens
        self._tokenizer: Any | None = None

    @property
    def max_content_tokens(self) -> int:
        return self._max

    def _ensure_tokenizer(self) -> Any:
        if self._tokenizer is None:
            self._tokenizer = self._factory()
        return self._tokenizer

    def split(self, text: str) -> tuple[WindowSpan, ...]:
        _validate_text(text)
        tokenizer = self._ensure_tokenizer()
        encoded = tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        offsets_raw = encoded["offset_mapping"]
        offsets: list[tuple[int, int]] = [
            (int(pair[0]), int(pair[1])) for pair in offsets_raw
        ]
        n_tokens = len(offsets)
        if n_tokens == 0:
            raise MetadataRAGWindowingError(
                "tokenizer produced zero tokens for a non-empty input"
            )

        boundaries: list[int] = [0]
        for tok_start in range(self._max, n_tokens, self._max):
            boundaries.append(offsets[tok_start][0])
        boundaries.append(len(text))

        cleaned: list[int] = [boundaries[0]]
        for boundary in boundaries[1:]:
            if boundary > cleaned[-1]:
                cleaned.append(boundary)

        if len(cleaned) < 2:
            raise MetadataRAGWindowingError(
                "windowing produced fewer than two distinct "
                "boundaries for a non-empty input"
            )

        spans: list[WindowSpan] = []
        for i in range(len(cleaned) - 1):
            spans.append(
                WindowSpan(
                    window_index=i,
                    char_start=cleaned[i],
                    char_end=cleaned[i + 1],
                )
            )
        return tuple(spans)


def build_bge_m3_window_splitter(
    *,
    snapshot_path: str | Path,
    max_content_tokens: int = WINDOW_CONTENT_TOKENS,
) -> BgeM3WindowSplitter:
    """Build a splitter that loads the pinned BGE-M3 tokenizer.

    The tokenizer is loaded lazily and offline from a local snapshot
    directory. No network access is performed at any point.
    """

    def _factory() -> Any:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(
            str(snapshot_path),
            local_files_only=True,
            use_fast=True,
        )

    return BgeM3WindowSplitter(
        tokenizer_factory=_factory,
        max_content_tokens=max_content_tokens,
    )
