"""Tests for native_e2e.runner (two-step generation, mock reader)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from baselines.base import RetrievedChunk
from baselines.longrag_v2.native_e2e.prompt_loader import load_prompt
from baselines.longrag_v2.native_e2e.runner import E2EResult, E2ERunner


class _MockBaseline:
    name = "mock-baseline"
    version = "0.0.1"

    def __init__(self, results_map: dict[str, list[RetrievedChunk]]) -> None:
        self._map = results_map
        self.retrieve_calls: list[tuple[str, int | None]] = []

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        self.retrieve_calls.append((query, top_k))
        return list(self._map.get(query, []))


class _MockReader:
    """Returns a different answer per call so the two-step flow is visible."""

    def __init__(self) -> None:
        self.prompts_seen: list[str] = []

    @property
    def name(self) -> str:
        return "mock-reader"

    def generate(self, prompt: str) -> str:
        self.prompts_seen.append(prompt)
        n = len(self.prompts_seen)
        return f"CALL_{n}"


def _chunk(doc_id: str, text: str, rank: int = 1, score: float = 1.0,
           title: str | None = None) -> RetrievedChunk:
    md: dict = {"unit_id": f"unit-{doc_id}"}
    if title:
        md["title"] = title
    return RetrievedChunk(
        doc_id=doc_id, chunk_id=doc_id, text=text, score=score, rank=rank,
        metadata=md,
    )


def _make_runner(baseline, reader, demos: str = "DEMOS\n\n") -> E2ERunner:
    return E2ERunner(
        baseline=baseline,
        reader=reader,
        main_prompt=load_prompt("nq"),
        extraction_prompt=load_prompt("answer_extraction"),
        demos=demos,
        dataset_name="nq",
    )


def test_two_step_invokes_reader_twice() -> None:
    baseline = _MockBaseline({"Q?": [_chunk("d1", "ctx")]})
    reader = _MockReader()
    runner = _make_runner(baseline, reader)
    r = runner.run_query("q1", "Q?")
    assert len(reader.prompts_seen) == 2
    # Step 1 output is long_answer; step 2 output is short_answer
    assert r.long_answer == "CALL_1"
    assert r.short_answer == "CALL_2"


def test_main_prompt_contains_context_and_titles() -> None:
    baseline = _MockBaseline({
        "Q?": [_chunk("d1", "hello", title="TitleA")]
    })
    reader = _MockReader()
    runner = _make_runner(baseline, reader)
    r = runner.run_query("q1", "Q?")
    assert "hello" in r.main_prompt
    assert "Q?" in r.main_prompt
    assert "TitleA" in r.main_prompt


def test_extraction_prompt_contains_long_answer_and_demos() -> None:
    baseline = _MockBaseline({"Q?": [_chunk("d1", "ctx")]})
    reader = _MockReader()
    runner = _make_runner(baseline, reader, demos="MY_DEMO_BLOCK\n\n")
    r = runner.run_query("q1", "Q?")
    # Step 2 prompt must contain the long answer from step 1
    assert "CALL_1" in r.extraction_prompt
    # And the provided demos
    assert "MY_DEMO_BLOCK" in r.extraction_prompt
    # And the question
    assert "Q?" in r.extraction_prompt


def test_no_retrieved() -> None:
    baseline = _MockBaseline({})
    reader = _MockReader()
    runner = _make_runner(baseline, reader)
    r = runner.run_query("q1", "Q?")
    assert r.context == ""
    assert r.titles == []
    assert r.num_unique_units == 0
    assert r.long_answer == "CALL_1"
    assert r.short_answer == "CALL_2"


def test_run_batch_with_tuples() -> None:
    baseline = _MockBaseline({"A": [_chunk("d1", "x")], "B": [_chunk("d2", "y")]})
    reader = _MockReader()
    runner = _make_runner(baseline, reader)
    results = runner.run_batch([("q1", "A"), ("q2", "B")])
    assert [r.query_id for r in results] == ["q1", "q2"]
    # 4 calls (2 per query)
    assert len(reader.prompts_seen) == 4


def test_run_batch_rejects_bad_query() -> None:
    baseline = _MockBaseline({})
    reader = _MockReader()
    runner = _make_runner(baseline, reader)
    with pytest.raises(TypeError):
        runner.run_batch([42])


def test_top_k_forwarded() -> None:
    baseline = _MockBaseline({"Q?": [_chunk("d1", "x")]})
    reader = _MockReader()
    runner = _make_runner(baseline, reader)
    runner.run_query("q1", "Q?", top_k=7)
    assert baseline.retrieve_calls == [("Q?", 7)]


def test_write_manifest(tmp_path: Path) -> None:
    baseline = _MockBaseline({"Q?": [_chunk("d1", "hello")]})
    reader = _MockReader()
    runner = _make_runner(baseline, reader)
    results = runner.run_batch([("q1", "Q?")])
    out = tmp_path / "m" / "manifest.json"
    runner.write_manifest(results, out, extra_meta={"dataset": "mock"})
    assert out.exists()
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["schema_version"] == 2
    assert data["dataset_name"] == "nq"
    assert data["main_prompt"]["name"] == "nq"
    assert data["extraction_prompt"]["name"] == "answer_extraction"
    assert "demos" in data
    assert data["num_queries"] == 1
    q = data["queries"][0]
    assert q["query_id"] == "q1"
    assert q["long_answer"] == "CALL_1"
    assert q["short_answer"] == "CALL_2"
    assert q["retrieved_doc_ids"] == ["d1"]
    assert "main_prompt_sha256" in q
    assert "extraction_prompt_sha256" in q


def test_sha256_fields_are_hex() -> None:
    baseline = _MockBaseline({"Q?": [_chunk("d1", "hello")]})
    reader = _MockReader()
    runner = _make_runner(baseline, reader)
    r = runner.run_query("q1", "Q?")
    assert len(r.main_prompt_sha256) == 64
    assert len(r.extraction_prompt_sha256) == 64
    assert len(r.context_sha256) == 64