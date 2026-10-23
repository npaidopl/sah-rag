"""BM25 baseline adapter for SAH-RAG v2.

Official BM25 from Pyserini/Anserini (Lucene backend), used as the lexical
baseline in the SAH-RAG comparison. Native reference:

    Lin, Ma, Lin, Yang, Pradeep, Nogueira.
    "Pyserini: An Easy-to-Use Python Toolkit to Support Replicable IR Research
     with Sparse and Dense Representations." arXiv:2102.10073.

Structural exception (documented, intentional):
    BM25 is lexical / bag-of-words. It has NO dense embedder. In controlled
    mode the orchestrator may inject `embedder` and `llm` (the frozen
    BaseBaseline signature requires accepting them), but BM25 IGNORES both.
    This is not a contract violation - it reflects a genuine architectural
    difference that must be declared in controlled-mode comparisons.

Chunking:
    BM25 operates at document level. `chunk_id == doc_id` by construction.

Performance note:
    The LuceneSearcher is lazily opened once on the first retrieve() and
    reused across calls. It is closed by cleanup() and at the start of a
    subsequent build_index() (dangling-ref safety). This avoids per-query
    index open/close overhead, which otherwise dominates runtime on
    benchmarks with hundreds/thousands of queries.
"""
from __future__ import annotations

import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from baselines.base import (
    BaseBaseline,
    BaselineConfig,
    BaselineStatus,
    Embedder,
    LLM,
    RetrievedChunk,
)

logger = logging.getLogger(__name__)


class BM25Baseline(BaseBaseline):
    """Pyserini/Anserini BM25, default Lucene BM25Similarity (k1=0.9, b=0.4)."""

    name = "bm25"
    version = "0.1.1"
    status = BaselineStatus.EXECUTABLE
    supports_native = True
    supports_controlled = True
    is_llm_based = False

    def __init__(
        self,
        *,
        config: BaselineConfig,
        work_dir: Path,
        embedder: Embedder | None = None,
        llm: LLM | None = None,
    ) -> None:
        super().__init__(config=config, work_dir=work_dir, embedder=embedder, llm=llm)
        self._index_dir = self.work_dir / "lucene_index"
        self._corpus_dir = self.work_dir / "corpus_jsonl"
        self._searcher: Any | None = None  # lazily opened in retrieve()
        self._k1 = float(self.config.extra.get("k1", 0.9))
        self._b = float(self.config.extra.get("b", 0.4))
        if self._k1 != 0.9 or self._b != 0.4:
            logger.warning(
                "BM25Baseline: k1/b overrides (%s, %s) are recorded but NOT applied; "
                "Pyserini CLI uses Lucene defaults (k1=0.9, b=0.4). "
                "Custom Similarity requires a Pyserini fork and is out of scope.",
                self._k1,
                self._b,
            )
        logger.info("BM25Baseline init: mode=%s index_dir=%s", config.mode.value, self._index_dir)

    # ----- lifecycle ------------------------------------------------------

    def build_index(self, documents: list[Any]) -> None:
        """Build a Lucene BM25 index from CanonicalDocument list.

        Writes CanonicalDocument -> JSONL (id, contents) under work_dir,
        then invokes the official Pyserini indexer CLI. Idempotent: closes
        any open searcher, rebuilds from scratch on each call.
        """
        if not documents:
            raise ValueError("BM25Baseline.build_index requires at least one document")

        # Close any open searcher before rebuilding (dangling-ref safety).
        self._close_searcher()

        if self._corpus_dir.exists():
            shutil.rmtree(self._corpus_dir)
        self._corpus_dir.mkdir(parents=True, exist_ok=True)

        corpus_jsonl = self._corpus_dir / "corpus.jsonl"
        with corpus_jsonl.open("w", encoding="utf-8") as f:
            for doc in documents:
                text = getattr(doc, "text_raw", "") or getattr(doc, "text_normalized", "") or ""
                record = {"id": str(doc.doc_id), "contents": text}
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        if self._index_dir.exists():
            shutil.rmtree(self._index_dir)
        self._index_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable, "-m", "pyserini.index.lucene",
            "--collection", "JsonCollection",
            "--input", str(self._corpus_dir),
            "--index", str(self._index_dir),
            "--generator", "DefaultLuceneDocumentGenerator",
            "--threads", "1",
            "--storePositions", "--storeDocvectors", "--storeRaw",
        ]
        logger.info("BM25Baseline: building index -> %s", self._index_dir)
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                "Pyserini indexer failed (rc="
                f"{result.returncode}).\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
            )
        logger.info("BM25Baseline: index ready at %s", self._index_dir)

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        """Return top-k BM25 hits with contiguous ranks 1..len(results)."""
        from pyserini.search.lucene import LuceneSearcher  # deferred: JVM startup

        k = int(top_k) if top_k is not None else int(self.config.top_k)
        if k <= 0:
            return []
        if not self._index_dir.exists():
            raise RuntimeError(
                f"BM25 index not found at {self._index_dir}. Call build_index() first."
            )

        # Lazy-open searcher ONCE and reuse across queries (perf-critical).
        if self._searcher is None:
            self._searcher = LuceneSearcher(str(self._index_dir))

        hits = self._searcher.search(query, k=k)
        results: list[RetrievedChunk] = []
        for rank, hit in enumerate(hits, start=1):
            raw = self._searcher.doc(hit.docid).raw()
            try:
                text = json.loads(raw).get("contents", "") or raw
            except Exception:
                text = raw
            results.append(
                RetrievedChunk(
                    doc_id=hit.docid,
                    chunk_id=hit.docid,  # BM25 has no chunking
                    text=text,
                    score=float(hit.score),
                    rank=rank,
                    parent_doc_id=None,
                    metadata={"retriever": "bm25", "mode": self.config.mode.value},
                )
            )
        return results

    def cleanup(self) -> None:
        """Close the Lucene searcher. Called after evaluation."""
        self._close_searcher()

    # ----- internal -------------------------------------------------------

    def _close_searcher(self) -> None:
        """Best-effort searcher close. Idempotent."""
        if self._searcher is not None:
            try:
                self._searcher.close()
            except Exception:
                pass
            self._searcher = None