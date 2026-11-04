"""Native end-to-end runner for LongRAG (retrieval + two-step reader).

Two-step flow (matching utils/gpt_inference.py at pinned commit):

    Step 1 (main prompt):
        prompt = render(main_prompt, titles=titles, context=context, question=q)
        long_answer = reader.generate(prompt)
        # reader.generate() applies post_process (extract after "answer:")

    Step 2 (extraction prompt):
        extract_prompt = render(extraction_prompt,
                                demos=demos,
                                question=q,
                                long_answer=long_answer)
        short_answer = reader.generate(extract_prompt)
        # reader.generate() applies post_process again

    The short_answer is the quantity scored against ground truth
    (exact-match per utils/eval_util.py).

The runner does not build the index. Callers invoke ``baseline.build_index``
themselves so index construction can be cached across runs.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from baselines.base import BaseBaseline, RetrievedChunk
from baselines.longrag_v2.native_e2e.context_builder import build_context
from baselines.longrag_v2.native_e2e.prompt_loader import (
    PromptTemplate,
    load_demo_examples,
    render,
)
from baselines.longrag_v2.native_e2e.reader.base import Reader

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@dataclass
class E2EResult:
    """Full record of one end-to-end run for a single query.

    Includes both intermediate artifacts (long_answer, prompts) and
    the scored output (short_answer).
    """

    query_id: str
    query: str
    # Two-step outputs:
    long_answer: str
    short_answer: str
    # Artifacts:
    context: str
    titles: list[str]
    main_prompt: str
    extraction_prompt: str
    retrieved: list[RetrievedChunk]
    # Provenance:
    reader_name: str
    main_prompt_sha256: str
    extraction_prompt_sha256: str
    context_sha256: str
    duration_ms: float
    num_unique_units: int
    extra: dict[str, Any] = field(default_factory=dict)


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
    """Compose retrieval + context building + two-step reader.

    Parameters
    ----------
    baseline
        A built BaseBaseline (``build_index`` already called).
    reader
        A Reader instance.
    main_prompt
        PromptTemplate for step 1 (must contain {context}, {question}, {titles}).
    extraction_prompt
        PromptTemplate for step 2 (must contain {demos}, {question}, {long_answer}).
    demos
        Few-shot demo block. When None, ``load_demo_examples()`` is used.
    max_context_chars
        Context truncation budget.
    dataset_name
        Free-form identifier (e.g. "nq", "hotpotqa"). Stored in manifests.
    """

    def __init__(
        self,
        *,
        baseline: BaseBaseline,
        reader: Reader,
        main_prompt: PromptTemplate,
        extraction_prompt: PromptTemplate,
        demos: str | None = None,
        max_context_chars: int = 120_000,
        dataset_name: str = "unknown",
    ) -> None:
        self._baseline = baseline
        self._reader = reader
        self._main_prompt = main_prompt
        self._extraction_prompt = extraction_prompt
        self._demos = demos if demos is not None else load_demo_examples()
        self._max_context_chars = max_context_chars
        self._dataset_name = dataset_name

    def run_query(
        self,
        query_id: str,
        query: str,
        top_k: int | None = None,
    ) -> E2EResult:
        t0 = time.perf_counter()

        # --- Retrieve ---
        retrieved = self._baseline.retrieve(query, top_k=top_k)

        # --- Build context + titles ---
        built = build_context(retrieved, max_chars=self._max_context_chars)

        # --- Step 1: main prompt ---
        main_prompt_text = render(
            self._main_prompt,
            titles=built.titles,
            context=built.context,
            question=query,
        )
        long_answer = self._reader.generate(main_prompt_text)

        # --- Step 2: extraction prompt ---
        extraction_prompt_text = render(
            self._extraction_prompt,
            demos=self._demos,
            question=query,
            long_answer=long_answer,
        )
        short_answer = self._reader.generate(extraction_prompt_text)

        duration_ms = (time.perf_counter() - t0) * 1000.0

        return E2EResult(
            query_id=str(query_id),
            query=str(query),
            long_answer=long_answer,
            short_answer=short_answer,
            context=built.context,
            titles=list(built.titles),
            main_prompt=main_prompt_text,
            extraction_prompt=extraction_prompt_text,
            retrieved=retrieved,
            reader_name=self._reader.name,
            main_prompt_sha256=hashlib.sha256(
                main_prompt_text.encode("utf-8")
            ).hexdigest(),
            extraction_prompt_sha256=hashlib.sha256(
                extraction_prompt_text.encode("utf-8")
            ).hexdigest(),
            context_sha256=hashlib.sha256(
                built.context.encode("utf-8")
            ).hexdigest(),
            duration_ms=duration_ms,
            num_unique_units=built.num_unique_units,
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
            "schema_version": 2,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "dataset_name": self._dataset_name,
            "baseline": {"name": baseline_name, "version": baseline_version},
            "reader": {"name": self._reader.name},
            "main_prompt": {
                "name": self._main_prompt.name,
                "sha256": self._main_prompt.sha256,
            },
            "extraction_prompt": {
                "name": self._extraction_prompt.name,
                "sha256": self._extraction_prompt.sha256,
            },
            "demos": {
                "length_chars": len(self._demos),
                "sha256": hashlib.sha256(
                    self._demos.encode("utf-8")
                ).hexdigest(),
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
                    "long_answer": r.long_answer,
                    "short_answer": r.short_answer,
                    "context_sha256": r.context_sha256,
                    "main_prompt_sha256": r.main_prompt_sha256,
                    "extraction_prompt_sha256": r.extraction_prompt_sha256,
                    "num_unique_units": r.num_unique_units,
                    "num_titles": len(r.titles),
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