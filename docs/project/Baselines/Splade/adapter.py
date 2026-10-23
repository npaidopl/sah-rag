"""SPLADE baseline adapter for SAH-RAG v2.

Native mode:
    Official SPLADE-v3 (naver/splade-v3) via Hugging Face transformers.
    Sparse encoding = ReLU -> log1p -> max-pool over tokens.
    Declared as AUTHOR_SUPPORTED_NATIVE (not PAPER_EXACT_NATIVE).

Controlled mode:
    Injected embedder (BGE-M3) -> dense vector -> top-M sparsification.
    Inverted index + sparse dot-product scoring preserved.
    Declared as controlled SPLADE approximation, not native SPLADE.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from baselines.base import (
    BaseBaseline,
    BaselineConfig,
    BaselineStatus,
    RetrievedChunk,
    RunMode,
)

logger = logging.getLogger(__name__)


class SpladeBaseline(BaseBaseline):
    """SPLADE sparse retrieval baseline (native + controlled)."""

    name = "splade"
    version = "0.1.0"
    status = BaselineStatus.EXECUTABLE
    supports_native = True
    supports_controlled = True
    is_llm_based = False

    def __init__(
        self,
        *,
        config: BaselineConfig,
        work_dir: Path,
        embedder: Any | None = None,
        llm: Any | None = None,
    ) -> None:
        super().__init__(
            config=config, work_dir=work_dir, embedder=embedder, llm=llm
        )
        self._model: Any | None = None
        self._tokenizer: Any | None = None
        self._index: dict[str, dict[int, float]] = {}
        self._inverted: dict[int, list[tuple[str, float]]] = {}
        self._doc_meta: dict[str, dict[str, Any]] = {}
        self._try_load_existing()

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def _try_load_existing(self) -> None:
        index_path = self.work_dir / "index.json"
        meta_path = self.work_dir / "doc_meta.json"
        if not (index_path.exists() and meta_path.exists()):
            return
        try:
            with index_path.open("r", encoding="utf-8") as f:
                raw = json.load(f)
            with meta_path.open("r", encoding="utf-8") as f:
                self._doc_meta = json.load(f)
            self._index = {
                doc_id: {int(d): float(w) for d, w in sparse.items()}
                for doc_id, sparse in raw.items()
            }
            self._inverted = {}
            for doc_id, sparse in self._index.items():
                for dim, w in sparse.items():
                    self._inverted.setdefault(dim, []).append((doc_id, w))
        except (json.JSONDecodeError, ValueError, KeyError) as exc:
            logger.warning(
                "Failed to load existing index from %s: %s", self.work_dir, exc
            )
            self._index = {}
            self._inverted = {}
            self._doc_meta = {}

    def _persist(self) -> None:
        index_path = self.work_dir / "index.json"
        meta_path = self.work_dir / "doc_meta.json"
        with index_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    doc_id: {str(d): w for d, w in sparse.items()}
                    for doc_id, sparse in self._index.items()
                },
                f,
            )
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(self._doc_meta, f, ensure_ascii=False)

    # ------------------------------------------------------------------ #
    # Native encoder (lazy)
    # ------------------------------------------------------------------ #
    def _load_native_encoder(self) -> None:
        if self._model is not None:
            return
        try:
            from transformers import AutoModelForMaskedLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "Native SPLADE mode requires `torch` and `transformers`. "
                "Install them via workspace/splade_dev/requirements.txt."
            ) from exc

        import os

        model_name = self.config.extra.get("model_name", "naver/splade-v3")
        local_files_only = bool(
            self.config.extra.get("local_files_only", False)
        )
        token = (
            os.environ.get("HF_TOKEN")
            or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        )
        logger.info(
            "Loading SPLADE model %s (local_files_only=%s, token=%s)",
            model_name,
            local_files_only,
            "set" if token else "none",
        )

        kwargs: dict = {"local_files_only": local_files_only}
        if token:
            kwargs["token"] = token

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)
            self._model = AutoModelForMaskedLM.from_pretrained(model_name, **kwargs)
        except Exception as exc:
            if local_files_only:
                raise
            logger.warning(
                "Online load failed (%s: %s); retrying with local cache only...",
                type(exc).__name__,
                str(exc)[:160],
            )
            kwargs["local_files_only"] = True
            self._tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)
            self._model = AutoModelForMaskedLM.from_pretrained(model_name, **kwargs)

        self._model.eval()

    def _encode_native(self, text: str) -> dict[int, float]:
        self._load_native_encoder()
        import torch

        max_length = int(self.config.extra.get("max_length", 512))
        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )
        with torch.no_grad():
            logits = self._model(**inputs).logits
        activations = torch.log1p(torch.relu(logits))
        mask = inputs["attention_mask"].unsqueeze(-1)
        pooled = torch.max(activations * mask, dim=1).values.squeeze(0)
        nonzero = torch.nonzero(pooled, as_tuple=False).squeeze(-1)
        return {int(i): float(pooled[i]) for i in nonzero.tolist()}

    # ------------------------------------------------------------------ #
    # Controlled encoder
    # ------------------------------------------------------------------ #
    def _sparsify(self, dense: list[float]) -> dict[int, float]:
        arr = np.asarray(dense, dtype=np.float32)
        m = int(self.config.extra.get("sparse_top_m", 100))
        m = min(m, arr.size)
        if m <= 0:
            return {}
        top_idx = np.argpartition(np.abs(arr), -m)[-m:]
        return {int(i): float(arr[i]) for i in top_idx if arr[i] != 0.0}

    def _encode_controlled_batch(
        self, texts: list[str]
    ) -> list[dict[int, float]]:
        if self.embedder is None:
            raise RuntimeError("Controlled mode requires embedder injection")
        dense = self.embedder.encode_documents(texts)
        return [self._sparsify(v) for v in dense]

    def _encode_controlled_query(self, query: str) -> dict[int, float]:
        if self.embedder is None:
            raise RuntimeError("Controlled mode requires embedder injection")
        return self._sparsify(self.embedder.encode_query(query))

    # ------------------------------------------------------------------ #
    # Sparse duck-typing (controlled mode enhancement)
    # ------------------------------------------------------------------ #
    def _try_encode_sparse_from_embedder(
        self, texts: list[str]
    ) -> list[dict[int, float]] | None:
        """Try to obtain native sparse output from the injected embedder.

        Supports:
          - custom `encode_sparse(texts)` -> list[dict[int, float]]
          - FlagEmbedding style `encode(texts, return_sparse=True)` ->
            {"lexical_weights": [...]}

        Returns None if no sparse interface is available; callers fall back
        to `_encode_controlled_batch` (top-M sparsification).
        """
        if self.embedder is None:
            return None

        if hasattr(self.embedder, "encode_sparse"):
            try:
                raw = self.embedder.encode_sparse(texts)  # type: ignore[attr-defined]
                return [self._normalize_sparse(v) for v in raw]
            except Exception as exc:
                logger.warning(
                    "encode_sparse failed, falling back to top-M: %s", exc
                )

        if hasattr(self.embedder, "encode"):
            try:
                raw = self.embedder.encode(texts, return_sparse=True)  # type: ignore[attr-defined]
                if isinstance(raw, dict) and "lexical_weights" in raw:
                    return [self._normalize_sparse(v) for v in raw["lexical_weights"]]
            except Exception as exc:
                logger.warning(
                    "FlagEmbedding-style encode failed, falling back to top-M: %s",
                    exc,
                )

        return None

    @staticmethod
    def _normalize_sparse(raw: Any) -> dict[int, float]:
        """Normalize various sparse formats to dict[int, float]."""
        if isinstance(raw, dict):
            out: dict[int, float] = {}
            for k, v in raw.items():
                try:
                    out[int(k)] = float(v)
                except (ValueError, TypeError):
                    continue
            return out
        if isinstance(raw, (list, tuple)):
            out = {}
            for item in raw:
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    try:
                        out[int(item[0])] = float(item[1])
                    except (ValueError, TypeError):
                        continue
                elif isinstance(item, dict):
                    out.update(SpladeBaseline._normalize_sparse(item))
            return out
        return {}

    # ------------------------------------------------------------------ #
    # BaseBaseline lifecycle
    # ------------------------------------------------------------------ #
    def build_index(self, documents: list[Any]) -> None:
        texts = [d.text_normalized or d.text_raw for d in documents]
        if self.config.mode == RunMode.NATIVE:
            sparse_vecs = [self._encode_native(t) for t in texts]
        else:
            sparse_vecs = self._try_encode_sparse_from_embedder(texts)
            if sparse_vecs is None:
                sparse_vecs = self._encode_controlled_batch(texts)

        self._index = {}
        self._inverted = {}
        self._doc_meta = {}
        for doc, sparse in zip(documents, sparse_vecs):
            self._index[doc.doc_id] = sparse
            self._doc_meta[doc.doc_id] = {
                "text": doc.text_normalized or doc.text_raw,
                "dataset": doc.dataset,
                "title": doc.title,
            }
            for dim, w in sparse.items():
                self._inverted.setdefault(dim, []).append((doc.doc_id, w))

        self._persist()
        logger.info("Indexed %d documents", len(self._index))

    def retrieve(
        self, query: str, top_k: int | None = None
    ) -> list[RetrievedChunk]:
        if not self._index:
            return []
        k = top_k if top_k is not None else self.config.top_k
        if self.config.mode == RunMode.NATIVE:
            q_sparse = self._encode_native(query)
        else:
            q_sparse_list = self._try_encode_sparse_from_embedder([query])
            if q_sparse_list is not None:
                q_sparse = q_sparse_list[0]
            else:
                q_sparse = self._encode_controlled_query(query)

        scores: dict[str, float] = {}
        for dim, qw in q_sparse.items():
            for doc_id, dw in self._inverted.get(dim, []):
                scores[doc_id] = scores.get(doc_id, 0.0) + qw * dw
        if not scores:
            return []

        ranked = sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:k]
        out: list[RetrievedChunk] = []
        for rank, (doc_id, score) in enumerate(ranked, start=1):
            meta = self._doc_meta[doc_id]
            out.append(
                RetrievedChunk(
                    doc_id=doc_id,
                    chunk_id=doc_id,
                    text=meta["text"],
                    score=float(score),
                    rank=rank,
                    parent_doc_id=None,
                    metadata={
                        "dataset": meta["dataset"],
                        "title": meta["title"],
                    },
                )
            )
        return out