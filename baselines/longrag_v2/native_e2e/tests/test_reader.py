"""Offline tests for the native_e2e reader module."""

from __future__ import annotations

import os

import pytest

from baselines.longrag_v2.native_e2e.reader import (
    GPT4oReader,
    GeminiReader,
    LocalReader,
    Reader,
)
from baselines.longrag_v2.native_e2e.reader.base import Reader as ReaderBase


def test_reader_protocol_is_runtime_checkable() -> None:
    for cls in (GPT4oReader, GeminiReader, LocalReader):
        instance = cls.__new__(cls)
        # Protocol conformance is structural; check attributes exist.
        assert hasattr(cls, "generate")
        assert hasattr(cls, "name")


def test_reader_base_exported() -> None:
    assert Reader is ReaderBase


def test_gpt4o_reader_name() -> None:
    r = GPT4oReader(model="gpt-4o")
    assert r.name == "openai::gpt-4o"


def test_gemini_reader_name() -> None:
    r = GeminiReader(model="gemini-1.5-pro")
    assert r.name == "google::gemini-1.5-pro"


def test_local_reader_name() -> None:
    r = LocalReader(model_name="Qwen/Qwen2.5-0.5B-Instruct")
    assert r.name.startswith("local::")


def test_gpt4o_reader_without_api_key_raises(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    r = GPT4oReader()
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        r.generate("hello")


def test_gemini_reader_without_api_key_raises(monkeypatch) -> None:
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    r = GeminiReader()
    with pytest.raises(RuntimeError, match="GOOGLE_API_KEY"):
        r.generate("hello")


@pytest.mark.skipif(
    os.environ.get("LONGRAG_TEST_LOCAL_READER") != "1",
    reason="Set LONGRAG_TEST_LOCAL_READER=1 to load the local model.",
)
def test_local_reader_smoke() -> None:
    r = LocalReader(max_new_tokens=32)
    out = r.generate("Answer in one word: the capital of France is")
    assert isinstance(out, str)
    assert len(out) > 0
def test_gpt4o_post_process_extracts_after_answer() -> None:
    # Verbatim behavior from utils/gpt_inference.py at pinned commit.
    assert GPT4oReader.post_process(
        "The Eiffel Tower is in Paris. Answer: Paris"
    ) == "Paris"
    assert GPT4oReader.post_process("Answer: 42") == "42"
    # Note: the official regex uses \s (single whitespace) in the
    # lookbehind, so "answer:  seven" (two spaces) yields " seven"
    # (leading space preserved) -- this is verbatim native-faithful
    # behavior from utils/gpt_inference.py at pinned commit.
    assert GPT4oReader.post_process("answer:  seven") == " seven"
    # No "answer:" in text -> returns raw unchanged.
    assert GPT4oReader.post_process("Paris") == "Paris"
    # Empty string -> returns empty.
    assert GPT4oReader.post_process("") == ""


def test_gpt4o_reader_defaults_match_paper() -> None:
    # Defaults must match baseline_audit/longrag_native_reader_config.yaml.
    r = GPT4oReader()
    assert r._temperature == 0.0
    assert r._seed == 12345
    assert r._max_tokens == 1000
    assert r._retry == 3
    assert r._delay_seconds == 5.0
    assert r._azure_endpoint is None
    assert r._azure_api_version is None


def test_gemini_reader_defaults_match_paper() -> None:
    r = GeminiReader()
    assert r._temperature == 0.0
    assert r._max_output_tokens == 1000


def test_gpt4o_reader_accepts_azure_params() -> None:
    r = GPT4oReader(
        azure_endpoint="https://test.openai.azure.com",
        azure_api_version="2024-02-01",
    )
    assert r._azure_endpoint == "https://test.openai.azure.com"
    assert r._azure_api_version == "2024-02-01"