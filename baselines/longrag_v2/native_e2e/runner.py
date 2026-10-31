"""Native end-to-end runner for LongRAG (retrieval + reader)."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from baselines.base import BaseBaseline, RetrievedChunk
from baselines.longrag_v2.native_e2e.context_builder import build_context
from baselines.longrag_v2.native_e2e.prompt_loader import PromptTemplate, render
from baselines.longrag_v2.native_e2e.reader.base import Reader

if TYPE_CHECKING:
    from sah_rag.canonical.models import CanonicalQuery


@dataclass
class E2EResult:
    """The full record of one end-to-end run for a single query."""

    query_id: str
    query: str
    answer: str
    context: str
    prompt: str
    retrieved: list[RetrievedChunk]
    reader_name: str
    context_sha256: str
    prompt_sha256: str
    duration_ms: float
    num_unique_units: int


def _normalize_query(q: Any) -> tuple[str, str]:
    """Accept (query_id, query) tuples or CanonicalQuery-like objects."""
    if isinstance(q, tuple) and len(q) == 2:
        qid, qtext = q
        return str(qid), str(qtext)
    if hasattr(q, "query_id") and hasattr(q, "question"):
        return str(q.query_id), str(q.question)
    raise TypeError(
        f"Expected (query_id, query) tuple or CanonicalQuery-like object, "
        f"got {type(q).__name__}"
    )


class E2ERunner:
    """Compose retrieval + context construction + reader into an end-to-end run.

    The runner does not build the index. Callers invoke ``baseline.build_index``
    themselves before running, so index construction can be cached across
    multiple runs.
    """

    def __init__(
        self,
        *,
        baseline: BaseBaseline,
        reader: Reader,
        prompt: PromptTemplate,
        max_context_chars: int = 120_000,
    ) -> None:
        self._baseline = baseline
        self._reader = reader
        self._prompt = prompt
        self._max_context_chars = max_context_chars

    def run_query(
        self,
        query_id: str,
        query: str,
        top_k: int | None = None,
    ) -> E2EResult:
        t0 = time.perf_counter()

        retrieved = self._baseline.retrieve(query, top_k=top_k)
        context, n_units = build_context(
            retrieved, max_chars=self._max_context_chars
        )
        prompt = render(self._prompt, context=context, question=query)
        answer = self._reader.generate(prompt)

        duration_ms = (time.perf_counter() - t0) * 1000.0

        return E2EResult(
            query_id=str(query_id),
            query=str(query),
            answer=answer,
            context=context,
            prompt=prompt,
            retrieved=retrieved,
            reader_name=self._reader.name,
            context_sha256=hashlib.sha256(context.encode("utf-8")).hexdigest(),
            prompt_sha256=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            duration_ms=duration_ms,
            num_unique_units=n_units,
        )

    def run_batch(
        self,
        queries: list[Any],
        top_k: int | None = None,
    ) -> list[E2EResult]:
        results: list[E2EResult] = []
        for q in queries:
            qid, qtext = _normalize_query(q)
            results.append(self.run_query(qid, qtext, top_k=top_k))
        return results

    def write_manifest(
        self,
        results: list[E2EResult],
        path: Path,
        *,
        extra_meta: dict[str, Any] | None = None,
    ) -> None:
        """Write a provenance manifest (JSON) under ``path``."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        baseline_name = getattr(self._baseline, "name", "unknown")
        baseline_version = getattr(self._baseline, "version", "unknown")

        manifest: dict[str, Any] = {
            "schema_version": 1,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "baseline": {"name": baseline_name, "version": baseline_version},
            "reader": {"name": self._reader.name},
            "prompt": {
                "name": self._prompt.name,
                "sha256": self._prompt.sha256,
            },
            "config": {
                "max_context_chars": self._max_context_chars,
            },
            "extra_meta": extra_meta or {},
            "num_queries": len(results),
            "queries": [
                {
                    "query_id": r.query_id,
                    "query": r.query,
                    "answer": r.answer,
                    "context_sha256": r.context_sha256,
                    "prompt_sha256": r.prompt_sha256,
                    "num_unique_units": r.num_unique_units,
                    "duration_ms": r.duration_ms,
                    "retrieved_doc_ids": [c.doc_id for c in r.retrieved],
                    "retrieved_scores": [c.score for c in r.retrieved],
                    "retrieved_unit_ids": [
                        (c.metadata or {}).get("unit_id") for c in r.retrieved
                    ],
                }
                for r in results
            ],
        }

        path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )