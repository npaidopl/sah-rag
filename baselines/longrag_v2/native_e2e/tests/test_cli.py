"""Tests for native_e2e CLI (loaders, builders, dry-run)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from baselines.base import RunMode
from baselines.longrag_v2.native_e2e.cli import (
    build_reader,
    load_baseline_config,
    load_documents_jsonl,
    load_queries_jsonl,
    main,
)
from baselines.longrag_v2.native_e2e.reader import (
    GPT4oReader,
    GeminiReader,
    LocalReader,
)


def _write_docs(path: Path, n: int = 3) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for i in range(n):
            f.write(json.dumps({
                "doc_id": f"d{i}", "dataset": "mock",
                "text_raw": f"document number {i}",
                "text_normalized": f"document number {i}",
            }) + "\n")
    return path


def _write_queries(path: Path, n: int = 2) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for i in range(n):
            f.write(json.dumps({
                "query_id": f"q{i}", "dataset": "mock",
                "question": f"question {i}?",
            }) + "\n")
    return path


def _write_yaml(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


@pytest.fixture()
def baseline_cfg_native(tmp_path: Path) -> Path:
    return _write_yaml(tmp_path / "baseline.yaml", """
name: longrag
mode: native
top_k: 5
extra:
  grouping_policy: one_document_per_unit
  target_unit_tokens: 4096
  k1: 100
""")


@pytest.fixture()
def reader_cfg_local(tmp_path: Path) -> Path:
    return _write_yaml(tmp_path / "reader.yaml", """
reader:
  type: local
  model: Qwen/Qwen2.5-0.5B-Instruct
  max_new_tokens: 32
  device: cpu
""")


def test_load_documents_jsonl(tmp_path: Path) -> None:
    docs = load_documents_jsonl(_write_docs(tmp_path / "docs.jsonl", 3))
    assert len(docs) == 3


def test_load_documents_jsonl_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_documents_jsonl(tmp_path / "nope.jsonl")


def test_load_documents_jsonl_invalid(tmp_path: Path) -> None:
    p = tmp_path / "bad.jsonl"
    p.write_text('{"missing":"x"}\n', encoding="utf-8")
    with pytest.raises(ValueError):
        load_documents_jsonl(p)


def test_load_queries_jsonl(tmp_path: Path) -> None:
    q = load_queries_jsonl(_write_queries(tmp_path / "q.jsonl", 2))
    assert len(q) == 2


def test_load_baseline_config(baseline_cfg_native: Path) -> None:
    cfg = load_baseline_config(baseline_cfg_native)
    assert cfg.mode == RunMode.NATIVE
    assert cfg.top_k == 5


def test_load_baseline_config_top_k_override(baseline_cfg_native: Path) -> None:
    cfg = load_baseline_config(baseline_cfg_native, top_k_override=20)
    assert cfg.top_k == 20


def test_build_reader_local(reader_cfg_local: Path) -> None:
    r = build_reader(reader_cfg_local)
    assert isinstance(r, LocalReader)


def test_build_reader_gpt4o(tmp_path: Path) -> None:
    p = _write_yaml(tmp_path / "g.yaml", """
reader:
  type: gpt4o
  model: gpt-4o
""")
    r = build_reader(p)
    assert isinstance(r, GPT4oReader)
    assert r.name == "openai::gpt-4o"


def test_build_reader_gemini(tmp_path: Path) -> None:
    p = _write_yaml(tmp_path / "g.yaml", """
reader:
  type: gemini
  model: gemini-1.5-pro
""")
    r = build_reader(p)
    assert isinstance(r, GeminiReader)


def test_build_reader_unknown(tmp_path: Path) -> None:
    p = _write_yaml(tmp_path / "bad.yaml", """
reader:
  type: nope
""")
    with pytest.raises(ValueError, match="Unknown reader type"):
        build_reader(p)


def test_main_missing_args_returns_2() -> None:
    assert main([]) == 2


def test_main_dry_run_nq(tmp_path: Path, baseline_cfg_native: Path,
                          reader_cfg_local: Path) -> None:
    docs = _write_docs(tmp_path / "docs.jsonl", 3)
    queries = _write_queries(tmp_path / "q.jsonl", 2)
    out = tmp_path / "out"
    rc = main([
        "--docs", str(docs),
        "--queries", str(queries),
        "--baseline-config", str(baseline_cfg_native),
        "--reader-config", str(reader_cfg_local),
        "--prompt", "nq",
        "--out", str(out),
        "--dry-run",
    ])
    assert rc == 0
    assert not (out / "index").exists()


def test_main_rejects_controlled_mode(tmp_path: Path,
                                     reader_cfg_local: Path) -> None:
    docs = _write_docs(tmp_path / "docs.jsonl", 2)
    queries = _write_queries(tmp_path / "q.jsonl", 1)
    bad = _write_yaml(tmp_path / "controlled.yaml", """
name: longrag
mode: controlled
top_k: 5
extra:
  grouping_policy: one_document_per_unit
""")
    rc = main([
        "--docs", str(docs),
        "--queries", str(queries),
        "--baseline-config", str(bad),
        "--reader-config", str(reader_cfg_local),
        "--prompt", "nq",
        "--out", str(tmp_path / "out"),
    ])
    assert rc == 1