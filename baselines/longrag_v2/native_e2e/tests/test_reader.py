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