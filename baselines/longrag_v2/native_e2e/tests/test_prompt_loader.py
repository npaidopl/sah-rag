"""Offline tests for native_e2e.prompt_loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from baselines.longrag_v2.native_e2e.prompt_loader import load_prompt, render


def test_load_by_name() -> None:
    p = load_prompt("nq")
    assert p.name == "nq"
    assert "{context}" in p.text
    assert "{question}" in p.text
    assert len(p.sha256) == 64


@pytest.mark.parametrize("name", ["nq", "hotpotqa", "qasper", "multifieldqa"])
def test_all_named_prompts_load(name: str) -> None:
    p = load_prompt(name)
    assert p.name == name
    assert p.sha256


def test_load_by_explicit_path(tmp_path: Path) -> None:
    f = tmp_path / "custom.txt"
    f.write_text("C: {context}\nQ: {question}", encoding="utf-8")
    p = load_prompt(f)
    assert p.name == "custom"
    assert p.source_path == f


def test_unknown_name_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_prompt("this_does_not_exist")


def test_render_substitutes() -> None:
    p = load_prompt("nq")
    out = render(p, context="MY_CONTEXT", question="MY_QUESTION?")
    assert "MY_CONTEXT" in out
    assert "MY_QUESTION?" in out
    assert "{context}" not in out
    assert "{question}" not in out


def test_render_missing_context_placeholder(tmp_path: Path) -> None:
    f = tmp_path / "bad.txt"
    f.write_text("Only {question}", encoding="utf-8")
    p = load_prompt(f)
    with pytest.raises(ValueError, match="context"):
        render(p, context="x", question="y")


def test_render_missing_question_placeholder(tmp_path: Path) -> None:
    f = tmp_path / "bad.txt"
    f.write_text("Only {context}", encoding="utf-8")
    p = load_prompt(f)
    with pytest.raises(ValueError, match="question"):
        render(p, context="x", question="y")