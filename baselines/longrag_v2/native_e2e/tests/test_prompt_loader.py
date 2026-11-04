"""Tests for native_e2e.prompt_loader (byte-faithful prompts)."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from baselines.longrag_v2.native_e2e.prompt_loader import (
    load_demo_examples,
    load_prompt,
    render,
)


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def test_load_nq() -> None:
    p = load_prompt("nq")
    assert p.name == "nq"
    assert "{context}" in p.text
    assert "{question}" in p.text
    assert "{titles}" in p.text
    assert len(p.sha256) == 64


def test_load_hotpotqa() -> None:
    p = load_prompt("hotpotqa")
    assert p.name == "hotpotqa"
    assert "{context}" in p.text
    assert "{question}" in p.text
    assert "{titles}" in p.text


def test_load_answer_extraction() -> None:
    p = load_prompt("answer_extraction")
    assert p.name == "answer_extraction"
    assert "{demos}" in p.text
    assert "{question}" in p.text
    assert "{long_answer}" in p.text


def test_qasper_is_gone() -> None:
    with pytest.raises(FileNotFoundError):
        load_prompt("qasper")


def test_multifieldqa_is_gone() -> None:
    with pytest.raises(FileNotFoundError):
        load_prompt("multifieldqa")


def test_unknown_name_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_prompt("does_not_exist")


# ---------------------------------------------------------------------------
# Exact-content checks (byte-faithful)
# ---------------------------------------------------------------------------

def test_nq_byte_faithful_ends_with_period_space() -> None:
    p = load_prompt("nq")
    # Official f-string ends with ". " (period + space), no newline
    assert p.text.endswith(". ")
    assert not p.text.endswith("\n")


def test_hotpotqa_byte_faithful_ends_with_two_spaces() -> None:
    p = load_prompt("hotpotqa")
    # Official f-string ends with "thing.  " (two spaces)
    assert p.text.endswith("thing.  ")
    assert not p.text.endswith("\n")


def test_hotpotqa_missing_space_after_period_preserved() -> None:
    # Official has "...question: {question}." directly followed by "For general..."
    # with NO space after the period. Must be preserved.
    p = load_prompt("hotpotqa")
    assert "{question}.For general" in p.text


def test_extract_missing_space_after_words_preserved() -> None:
    # Official has "...redundant words.It's important..." (typo preserved)
    p = load_prompt("answer_extraction")
    assert "words.It's important" in p.text


def test_extract_ends_with_short_answer_space() -> None:
    p = load_prompt("answer_extraction")
    assert p.text.endswith("Short Answer: ")


# ---------------------------------------------------------------------------
# Render (single-stage)
# ---------------------------------------------------------------------------

def test_render_nq_with_titles() -> None:
    p = load_prompt("nq")
    out = render(
        p,
        titles=["Eiffel Tower", "Louvre"],
        context="CTX",
        question="Q?",
    )
    assert "CTX" in out
    assert "Q?" in out
    # Titles formatted as list of double-quoted strings
    assert "['\"Eiffel Tower\"', '\"Louvre\"']" in out
    assert "{titles}" not in out
    assert "{context}" not in out
    assert "{question}" not in out


def test_render_nq_missing_titles_raises() -> None:
    p = load_prompt("nq")
    with pytest.raises(ValueError, match="unresolved placeholders"):
        render(p, context="x", question="y")


def test_render_hotpotqa_full() -> None:
    p = load_prompt("hotpotqa")
    out = render(
        p,
        titles=["A", "B"],
        context="C",
        question="Q",
    )
    assert "C" in out and "Q" in out
    assert "A" in out and "B" in out


# ---------------------------------------------------------------------------
# Render (two-stage extraction)
# ---------------------------------------------------------------------------

def test_render_extract_full() -> None:
    p = load_prompt("answer_extraction")
    out = render(
        p,
        demos="Here are some examples: DEMO1 DEMO2\n\n",
        question="Q?",
        long_answer="LA",
    )
    assert "DEMO1" in out
    assert "LA" in out
    assert "Q?" in out
    assert "Short Answer:" in out


def test_render_extract_missing_demos_raises() -> None:
    p = load_prompt("answer_extraction")
    with pytest.raises(ValueError, match="unresolved placeholders"):
        render(p, question="Q", long_answer="LA")


# ---------------------------------------------------------------------------
# Demo examples
# ---------------------------------------------------------------------------

def test_load_demo_examples_returns_string() -> None:
    demos = load_demo_examples()
    assert isinstance(demos, str)
    assert len(demos) > 0


def test_demo_examples_structure() -> None:
    demos = load_demo_examples()
    # Each demo block has Question/Long Answer/Short Answer labels
    # (unless placeholder, which still has them)
    assert "Question:" in demos
    assert "Long Answer:" in demos
    assert "Short Answer:" in demos