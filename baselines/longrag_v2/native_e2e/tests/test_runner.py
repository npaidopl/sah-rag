"""Offline tests for native_e2e.runner (end-to-end with mock reader)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from baselines.base import RetrievedChunk
from baselines.longrag_v2.native_e2e.prompt_loader import load_prompt
from baselines.longrag_v2.native_e2e.runner import E2EResult, E2ERunner


class _MockBaseline:
    """Duck-typed stand-in for BaseBaseline (only retrieve is used)."""

    name = "mock-baseline"
    version = "0.0.1"

    def __init__(self, results_map: dict[str, list[RetrievedChunk]]) -> None:
        self._map = results_map
        self.retrieve_calls: list[tuple[str, int | None]] = []

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        self.retrieve_calls.append((query, top_k))
        return list(self._map.get(query, []))


class _MockReader:
    def __init__(self, answer: str = "MOCK") -> None:
        self._answer = answer
        self.prompts_seen: list[str] = []

    @property
    def name(self) -> str:
        return "mock-reader"

    def generate(self, prompt: str) -> str:
        self.prompts_seen.append(prompt)
        return self._answer


def _chunk(doc_id: str, text: str, rank: int = 1, score: float = 1.0) -> RetrievedChunk:
    return RetrievedChunk(
        doc_id=doc_id,
        chunk_id=doc_id,
        text=text,
        score=score,
        rank=rank,
        metadata={"unit_id": f"unit-{doc_id}"},
    )


def test_run_query_basic() -> None:
    baseline = _MockBaseline({"Q?": [_chunk("d1", "hello world")]})
    reader = _MockReader(answer="A")
    runner = E2ERunner(
        baseline=baseline,  # type: ignore[arg-type]
        reader=reader,
        prompt=load_prompt("nq"),
    )
    r = runner.run_query("q1", "Q?")
    assert isinstance(r, E2EResult)
    assert r.query_id == "q1"
    assert r.query == "Q?"
    assert r.answer == "A"
    assert "hello world" in r.context
    assert "hello world" in r.prompt
    assert r.reader_name == "mock-reader"
    assert len(r.context_sha256) == 64
    assert len(r.prompt_sha256) == 64
    assert r.num_unique_units == 1
    assert r.duration_ms >= 0.0
    assert len(reader.prompts_seen) == 1


def test_run_query_no_retrieved() -> None:
    baseline = _MockBaseline({})
    reader = _MockReader()
    runner = E2ERunner(
        baseline=baseline,  # type: ignore[arg-type]
        reader=reader,
        prompt=load_prompt("nq"),
    )
    r = runner.run_query("q1", "Q?")
    assert r.context == ""
    assert r.num_unique_units == 0
    assert r.answer == "MOCK"


def test_run_batch_with_tuples() -> None:
    baseline = _MockBaseline({"A": [_chunk("d1", "x")], "B": [_chunk("d2", "y")]})
    reader = _MockReader()
    runner = E2ERunner(
        baseline=baseline,  # type: ignore[arg-type]
        reader=reader,
        prompt=load_prompt("nq"),
    )
    results = runner.run_batch([("q1", "A"), ("q2", "B")])
    assert [r.query_id for r in results] == ["q1", "q2"]
    assert len(reader.prompts_seen) == 2


def test_run_batch_with_canonical_query_like() -> None:
    class _CQ:
        query_id = "cq1"
        question = "Q?"

    baseline = _MockBaseline({"Q?": [_chunk("d1", "x")]})
    reader = _MockReader()
    runner = E2ERunner(
        baseline=baseline,  # type: ignore[arg-type]
        reader=reader,
        prompt=load_prompt("nq"),
    )
    results = runner.run_batch([_CQ()])
    assert results[0].query_id == "cq1"
    assert results[0].query == "Q?"


def test_run_batch_rejects_bad_query() -> None:
    baseline = _MockBaseline({})
    reader = _MockReader()
    runner = E2ERunner(
        baseline=baseline,  # type: ignore[arg-type]
        reader=reader,
        prompt=load_prompt("nq"),
    )
    with pytest.raises(TypeError):
        runner.run_batch([42])


def test_write_manifest(tmp_path: Path) -> None:
    baseline = _MockBaseline({"Q?": [_chunk("d1", "hello")]})
    reader = _MockReader(answer="Ans")
    runner = E2ERunner(
        baseline=baseline,  # type: ignore[arg-type]
        reader=reader,
        prompt=load_prompt("nq"),
    )
    results = runner.run_batch([("q1", "Q?")])
    out = tmp_path / "sub" / "manifest.json"
    runner.write_manifest(results, out, extra_meta={"dataset": "mock"})
    assert out.exists()

    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["schema_version"] == 1
    assert data["baseline"]["name"] == "mock-baseline"
    assert data["reader"]["name"] == "mock-reader"
    assert data["prompt"]["name"] == "nq"
    assert data["extra_meta"] == {"dataset": "mock"}
    assert data["num_queries"] == 1
    assert data["queries"][0]["query_id"] == "q1"
    assert data["queries"][0]["answer"] == "Ans"
    assert data["queries"][0]["retrieved_doc_ids"] == ["d1"]
    assert data["queries"][0]["retrieved_unit_ids"] == ["unit-d1"]


def test_top_k_forwarded() -> None:
    baseline = _MockBaseline({"Q?": [_chunk("d1", "x")]})
    reader = _MockReader()
    runner = E2ERunner(
        baseline=baseline,  # type: ignore[arg-type]
        reader=reader,
        prompt=load_prompt("nq"),
    )
    runner.run_query("q1", "Q?", top_k=7)
    assert baseline.retrieve_calls == [("Q?", 7)]