"""MemoRAG baseline adapter for SAH-RAG v2."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from baselines.base import (
    BaseBaseline,
    BaselineConfig,
    BaselineStatus,
    RetrievedChunk,
    RunMode,
)

logger = logging.getLogger(__name__)

DEFAULT_MEMORY_MODEL = "TommyChien/memorag-qwen2-7b-inst"
DEFAULT_RETRIEVER_MODEL = "BAAI/bge-m3"


class MemoRAGBaseline(BaseBaseline):
    """MemoRAG: memory-conditioned clue generation + retrieval."""

    name = "memorag"
    version = "0.1.0"
    status = BaselineStatus.EXECUTABLE
    supports_native = True
    supports_controlled = True
    is_llm_based = True

    def __init__(
        self,
        *,
        config: BaselineConfig,
        work_dir: Path,
        embedder=None,
        llm=None,
    ) -> None:
        super().__init__(
            config=config,
            work_dir=work_dir,
            embedder=embedder,
            llm=llm,
        )
        self._pipeline = None
        self._doc_lookup: dict[str, Any] = {}
        self._index_built = False
        self._memory_model = str(
            config.extra.get("memory_model", DEFAULT_MEMORY_MODEL)
        )
        self._retriever_model = str(
            config.extra.get("retriever_model", DEFAULT_RETRIEVER_MODEL)
        )
        if self.config.mode == RunMode.CONTROLLED and self.embedder is not None:
            logger.info(
                "Controlled mode; injected embedder=%s. "
                "MemoRAG architecture-defining components (memory model + "
                "clue generation) remain native.",
                getattr(self.embedder, "name", "unknown"),
            )

    def _ensure_pipeline(self):
        if self._pipeline is not None:
            return self._pipeline
        try:
            from memorag import MemoRAG  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "memorag package is not installed. "
                "Install with: pip install memorag"
            ) from exc
        cache_dir = self.work_dir / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        self._pipeline = MemoRAG(
            mem_model_name_or_path=self._memory_model,
            ret_model_name_or_path=self._retriever_model,
            cache_dir=str(cache_dir),
        )
        return self._pipeline

    def build_index(self, documents: list[Any]) -> None:
        self._doc_lookup = {doc.doc_id: doc for doc in documents}
        texts = [doc.text_normalized for doc in documents]
        pipeline = self._ensure_pipeline()
        load_fn = getattr(pipeline, "load", None)
        if load_fn is None:
            load_fn = getattr(pipeline, "load_corpus", None)
        if load_fn is None:
            raise RuntimeError(
                "MemoRAG pipeline exposes neither load() nor load_corpus()"
            )
        load_fn(texts)
        self._index_built = True
        logger.info(
            "MemoRAG indexed %d documents (mode=%s)",
            len(texts),
            self.config.mode.value,
        )

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        if not self._index_built:
            raise RuntimeError("build_index() must be called before retrieve()")
        k = top_k if top_k is not None else self.config.top_k
        pipeline = self._ensure_pipeline()
        try:
            raw = pipeline(query, task_type="retrieval", top_k=k)
        except TypeError:
            raw = pipeline(
                context="", query=query, task_type="retrieval", top_k=k
            )
        if raw is None:
            return []
        if isinstance(raw, dict):
            raw = raw.get("results") or raw.get("chunks") or [raw]
        chunks: list[RetrievedChunk] = []
        for item in raw:
            if isinstance(item, str):
                doc_id = item
                text = ""
                score = 1.0
            else:
                doc_id = str(
                    item.get("doc_id")
                    or item.get("id")
                    or item.get("pid")
                    or ""
                )
                text = str(item.get("text") or item.get("content") or "")
                score = float(item.get("score", 0.0))
            if not doc_id:
                continue
            if doc_id in self._doc_lookup and not text:
                text = self._doc_lookup[doc_id].text_normalized
            chunks.append(
                RetrievedChunk(
                    doc_id=doc_id,
                    chunk_id=f"{doc_id}::chunk_0",
                    text=text,
                    score=score,
                    rank=len(chunks) + 1,
                )
            )
            if len(chunks) >= k:
                break
        return chunks

    def cleanup(self) -> None:
        self._pipeline = None
        self._index_built = False
        self._doc_lookup = {}