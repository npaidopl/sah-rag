"""Tests for baselines.metadata_rag_windowing."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from baselines.metadata_rag_windowing import (
    MAXIMUM_MODEL_INPUT_TOKENS,
    SPECIAL_TOKEN_COUNT,
    WINDOW_CONTENT_TOKENS,
    WINDOWING_VERSION,
    BgeM3WindowSplitter,
    MetadataRAGWindowingError,
    SingleWindowSplitter,
    WindowSpan,
    WindowSplitterProtocol,
    build_bge_m3_window_splitter,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
AMENDMENT_04 = (
    REPO_ROOT
    / "baseline_audit"
    / "metadata_rag_clean_room_spec_amendment_04.yaml"
)


class _FakeWhitespaceTokenizer:
    """Deterministic whitespace tokenizer with character offsets.

    Emits one token per maximal run of non-whitespace characters.
    Offsets are (start, end) character positions in the input text.
    """

    def __init__(self) -> None:
        self.call_count = 0

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        return_offsets_mapping: bool,
        return_attention_mask: bool,
        return_token_type_ids: bool,
    ) -> dict[str, list[tuple[int, int]]]:
        del add_special_tokens, return_attention_mask
        del return_token_type_ids
        self.call_count += 1
        if not return_offsets_mapping:
            raise AssertionError("offset mapping must be requested")
        offsets: list[tuple[int, int]] = []
        i = 0
        n = len(text)
        while i < n:
            while i < n and text[i].isspace():
                i += 1
            if i >= n:
                break
            start = i
            while i < n and not text[i].isspace():
                i += 1
            offsets.append((start, i))
        return {"offset_mapping": offsets}


def _fixed_factory(
    tokenizer: Any,
) -> tuple[Any, list[int]]:
    counter = [0]

    def _factory() -> Any:
        counter[0] += 1
        return tokenizer

    return _factory, counter


# --- Constants match amendment 04 -----------------------------------------


def test_constants_match_amendment_04() -> None:
    data = yaml.safe_load(
        AMENDMENT_04.read_text(encoding="utf-8")
    )
    windowing = data["amendment"]["patch"]["metadata_schema"]["windowing"]
    assert windowing["window_content_tokens"] == WINDOW_CONTENT_TOKENS
    assert windowing["special_token_count"] == SPECIAL_TOKEN_COUNT
    assert (
        windowing["maximum_model_input_tokens"]
        == MAXIMUM_MODEL_INPUT_TOKENS
    )
    assert (
        WINDOW_CONTENT_TOKENS + SPECIAL_TOKEN_COUNT
        == MAXIMUM_MODEL_INPUT_TOKENS
    )


def test_windowing_version_is_exposed() -> None:
    assert WINDOWING_VERSION == "0.1.0"


# --- WindowSpan -----------------------------------------------------------


def test_window_span_accepts_valid_span() -> None:
    span = WindowSpan(window_index=0, char_start=0, char_end=10)
    assert span.char_start == 0
    assert span.char_end == 10


def test_window_span_rejects_zero_length() -> None:
    with pytest.raises(ValidationError, match="strictly greater"):
        WindowSpan(window_index=0, char_start=5, char_end=5)


def test_window_span_rejects_reversed() -> None:
    with pytest.raises(ValidationError, match="strictly greater"):
        WindowSpan(window_index=0, char_start=10, char_end=5)


def test_window_span_rejects_negative_start() -> None:
    with pytest.raises(ValidationError):
        WindowSpan(window_index=0, char_start=-1, char_end=10)


def test_window_span_rejects_negative_index() -> None:
    with pytest.raises(ValidationError):
        WindowSpan(window_index=-1, char_start=0, char_end=10)


def test_window_span_is_frozen() -> None:
    span = WindowSpan(window_index=0, char_start=0, char_end=10)
    with pytest.raises(ValidationError):
        span.char_start = 5


# --- SingleWindowSplitter --------------------------------------------------


def test_single_window_splitter_returns_one_window() -> None:
    splitter = SingleWindowSplitter()
    spans = splitter.split("hello world")
    assert len(spans) == 1
    assert spans[0].window_index == 0
    assert spans[0].char_start == 0
    assert spans[0].char_end == len("hello world")


def test_single_window_splitter_handles_long_text() -> None:
    splitter = SingleWindowSplitter()
    text = "x" * 100_000
    spans = splitter.split(text)
    assert len(spans) == 1
    assert spans[0].char_end == len(text)


def test_single_window_splitter_rejects_empty() -> None:
    with pytest.raises(MetadataRAGWindowingError, match="non-empty"):
        SingleWindowSplitter().split("")


def test_single_window_splitter_rejects_bad_max() -> None:
    with pytest.raises(MetadataRAGWindowingError, match="max_content"):
        SingleWindowSplitter(max_content_tokens=0)


def test_single_window_splitter_satisfies_protocol() -> None:
    assert isinstance(SingleWindowSplitter(), WindowSplitterProtocol)


# --- BgeM3WindowSplitter --------------------------------------------------


def test_bge_m3_splitter_short_text_one_window() -> None:
    tokenizer = _FakeWhitespaceTokenizer()
    factory, counter = _fixed_factory(tokenizer)
    splitter = BgeM3WindowSplitter(
        tokenizer_factory=factory, max_content_tokens=10
    )
    spans = splitter.split("one two three")
    assert len(spans) == 1
    assert spans[0].char_start == 0
    assert spans[0].char_end == len("one two three")
    assert counter[0] == 1


def test_bge_m3_splitter_exact_budget_one_window() -> None:
    tokenizer = _FakeWhitespaceTokenizer()
    factory, _ = _fixed_factory(tokenizer)
    splitter = BgeM3WindowSplitter(
        tokenizer_factory=factory, max_content_tokens=5
    )
    text = "one two three four five"
    spans = splitter.split(text)
    assert len(spans) == 1


def test_bge_m3_splitter_over_budget_splits() -> None:
    tokenizer = _FakeWhitespaceTokenizer()
    factory, _ = _fixed_factory(tokenizer)
    splitter = BgeM3WindowSplitter(
        tokenizer_factory=factory, max_content_tokens=2
    )
    text = "one two three four five"
    spans = splitter.split(text)
    assert len(spans) == 3
    assert [s.window_index for s in spans] == [0, 1, 2]


def test_bge_m3_splitter_windows_are_contiguous() -> None:
    tokenizer = _FakeWhitespaceTokenizer()
    factory, _ = _fixed_factory(tokenizer)
    splitter = BgeM3WindowSplitter(
        tokenizer_factory=factory, max_content_tokens=3
    )
    text = "alpha beta gamma delta epsilon zeta eta theta"
    spans = splitter.split(text)
    for left, right in zip(spans, spans[1:], strict=False):
        assert left.char_end == right.char_start


def test_bge_m3_splitter_windows_cover_entire_text() -> None:
    tokenizer = _FakeWhitespaceTokenizer()
    factory, _ = _fixed_factory(tokenizer)
    splitter = BgeM3WindowSplitter(
        tokenizer_factory=factory, max_content_tokens=2
    )
    text = "one two three four five six seven"
    spans = splitter.split(text)
    assert spans[0].char_start == 0
    assert spans[-1].char_end == len(text)
    reconstructed = "".join(
        text[s.char_start : s.char_end] for s in spans
    )
    assert reconstructed == text


def test_bge_m3_splitter_is_deterministic() -> None:
    tokenizer = _FakeWhitespaceTokenizer()
    factory, _ = _fixed_factory(tokenizer)
    splitter = BgeM3WindowSplitter(
        tokenizer_factory=factory, max_content_tokens=3
    )
    text = "a b c d e f g h i j"
    first = splitter.split(text)
    second = splitter.split(text)
    assert first == second


def test_bge_m3_splitter_rejects_empty_text() -> None:
    tokenizer = _FakeWhitespaceTokenizer()
    factory, _ = _fixed_factory(tokenizer)
    splitter = BgeM3WindowSplitter(
        tokenizer_factory=factory, max_content_tokens=10
    )
    with pytest.raises(MetadataRAGWindowingError, match="non-empty"):
        splitter.split("")


def test_bge_m3_splitter_rejects_zero_tokens() -> None:
    class _EmptyTokenizer:
        def __call__(self, text: str, **_kwargs: Any) -> dict[str, Any]:
            del text
            return {"offset_mapping": []}

    splitter = BgeM3WindowSplitter(
        tokenizer_factory=lambda: _EmptyTokenizer(),
        max_content_tokens=10,
    )
    with pytest.raises(
        MetadataRAGWindowingError, match="zero tokens"
    ):
        splitter.split("   ")


def test_bge_m3_splitter_rejects_bad_max() -> None:
    tokenizer = _FakeWhitespaceTokenizer()
    factory, _ = _fixed_factory(tokenizer)
    with pytest.raises(MetadataRAGWindowingError, match="max_content"):
        BgeM3WindowSplitter(
            tokenizer_factory=factory, max_content_tokens=0
        )


def test_bge_m3_splitter_rejects_non_callable_factory() -> None:
    with pytest.raises(
        MetadataRAGWindowingError, match="callable"
    ):
        BgeM3WindowSplitter(
            tokenizer_factory="not-callable",  # type: ignore[arg-type]
            max_content_tokens=10,
        )


def test_bge_m3_splitter_loads_tokenizer_lazily() -> None:
    tokenizer = _FakeWhitespaceTokenizer()
    factory, counter = _fixed_factory(tokenizer)
    splitter = BgeM3WindowSplitter(
        tokenizer_factory=factory, max_content_tokens=10
    )
    assert counter[0] == 0
    splitter.split("hello")
    assert counter[0] == 1
    splitter.split("hello again")
    assert counter[0] == 1


def test_bge_m3_splitter_satisfies_protocol() -> None:
    tokenizer = _FakeWhitespaceTokenizer()
    factory, _ = _fixed_factory(tokenizer)
    assert isinstance(
        BgeM3WindowSplitter(
            tokenizer_factory=factory, max_content_tokens=10
        ),
        WindowSplitterProtocol,
    )


# --- Production factory ----------------------------------------------------


def test_build_bge_m3_window_splitter_does_not_load_tokenizer() -> None:
    # Constructing the production splitter must not touch the network
    # or load any model. Only a real ``split`` call would do so.
    splitter = build_bge_m3_window_splitter(
        snapshot_path="C:/nonexistent/snapshot"
    )
    assert splitter.max_content_tokens == WINDOW_CONTENT_TOKENS
