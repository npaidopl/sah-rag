"""LongRAG baseline adapter for SAH-RAG v2 (IR-only).

Reference:
    Jiang et al. (2024), arXiv:2406.15319.
    Official repository: https://github.com/TIGER-AI-Lab/LongRAG (MIT).

Architecture-defining retrieval logic that this adapter preserves:

    * Long retrieval units (>4K tokens, grouped from related documents).
    * Top-k1 unit retrieval from the corpus, then expansion of each unit
      into its constituent documents (MaxP approximation, per the paper).
    * Return top-k documents after expansion.

Grouping policies implemented (selectable via ``extra.grouping_policy``):

    * "one_document_per_unit"      -- paper-faithful for non-Wikipedia tasks.
    * "adjacent_document_segments" -- the policy used by the existing
                                      controlled LongRAG baseline (L1b..L7).
    * "hyperlink_based"            -- paper-faithful for Wikipedia tasks;
                                      requires metadata["hyperlinks"] on each
                                      CanonicalDocument, otherwise falls back
                                      to one_document_per_unit.

The reader is intentionally omitted (v2 is IR-only).
"""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from baselines.base import (
    BaseBaseline,
    BaselineConfig,
    BaselineStatus,
    Embedder,
    LLM,
    RetrievedChunk,
    RunMode,
)
from sah_rag.canonical.models import CanonicalDocument

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Token estimation (approximate; no tokenizer dependency in controlled mode)
# ---------------------------------------------------------------------------

def _estimate_tokens(text: str) -> int:
    """Approximate token count for grouping decisions.

    We deliberately avoid loading a tokenizer here so that controlled
    mode stays tokenizer-agnostic (the injected Embedder protocol does
    not expose a tokenizer). The heuristic uses the maximum of the
    whitespace count and ceil(len/4), which is a reasonable lower
    bound for English text under BPE-like tokenizers.
    """
    if not text:
        return 0
    ws = len(text.split())
    ch = max(1, math.ceil(len(text) / 4.0))
    return max(ws, ch)


# ---------------------------------------------------------------------------
# Internal unit representation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _Unit:
    """A retrieval unit (LongRAG's "long unit")."""

    unit_id: str
    text: str
    doc_ids: tuple[str, ...]
    token_count: int
    title_map: dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Grouping policies
# ---------------------------------------------------------------------------

def _doc_text(doc: CanonicalDocument) -> str:
    """Preferred text field with fallback."""
    return doc.text_normalized or doc.text_raw or ""


def _doc_title(doc: CanonicalDocument) -> str:
    """Preferred title with fallback to doc_id.

    Resolution order:
        1. doc.title (CanonicalDocument top-level field)
        2. doc.metadata["title"] (canonical metadata)
        3. doc.doc_id (always present)
    """
    if doc.title and doc.title.strip():
        return doc.title.strip()
    if isinstance(doc.metadata, dict):
        t = doc.metadata.get("title")
        if isinstance(t, str) and t.strip():
            return t.strip()
    return doc.doc_id


def _group_one_document_per_unit(
    documents: list[CanonicalDocument],
) -> list[_Unit]:
    """Paper-faithful grouping for non-Wikipedia tasks.

    Each CanonicalDocument becomes exactly one retrieval unit.
    """
    units: list[_Unit] = []
    for doc in documents:
        text = _doc_text(doc)
        units.append(
            _Unit(
                unit_id=f"unit::{doc.doc_id}",
                text=text,
                doc_ids=(doc.doc_id,),
                token_count=_estimate_tokens(text),
                title_map={doc.doc_id: _doc_title(doc)},
            )
        )
    return units


def _group_adjacent_segments(
    documents: list[CanonicalDocument],
    target_tokens: int,
) -> list[_Unit]:
    """Pack consecutive documents into units up to ``target_tokens``.

    This is the policy used by the existing controlled LongRAG baseline
    (see baseline_audit/longrag_full_track_* contracts). It is an
    adaptation, not paper-faithful, but it is a reasonable default for
    heterogeneous non-Wikipedia corpora where individual documents may
    be much smaller or much larger than 4K tokens.

    Documents are processed in input order. A new unit is started
    whenever adding the next document would exceed ``target_tokens``.
    Oversized single documents are placed in their own unit.
    """
    units: list[_Unit] = []
    cur_texts: list[str] = []
    cur_doc_ids: list[str] = []
    cur_title_map: dict[str, str] = {}
    cur_tokens: int = 0
    seq: int = 0

    def flush() -> None:
        nonlocal cur_texts, cur_doc_ids, cur_title_map, cur_tokens, seq
        if not cur_texts:
            return
        units.append(
            _Unit(
                unit_id=f"unit::seg{seq:05d}",
                text="\n\n".join(cur_texts),
                doc_ids=tuple(cur_doc_ids),
                token_count=cur_tokens,
                title_map=dict(cur_title_map),
            )
        )
        seq += 1
        cur_texts = []
        cur_doc_ids = []
        cur_title_map = {}
        cur_tokens = 0

    for doc in documents:
        text = _doc_text(doc)
        toks = _estimate_tokens(text)
        if cur_texts and cur_tokens + toks > target_tokens:
            flush()
        cur_texts.append(text)
        cur_doc_ids.append(doc.doc_id)
        cur_title_map[doc.doc_id] = _doc_title(doc)
        cur_tokens += toks

    flush()
    return units


def _group_hyperlink_based(
    documents: list[CanonicalDocument],
    target_tokens: int,
) -> list[_Unit]:
    """Hyperlink-based grouping (paper-faithful for Wikipedia tasks).

    Requires ``metadata["hyperlinks"]`` on each CanonicalDocument to be a
    list of doc_ids that the document links to. Union-find clusters the
    documents; each cluster is then packed into adjacent segments up to
    ``target_tokens``, identical to ``_group_adjacent_segments`` applied
    per cluster.

    Falls back to ``_group_one_document_per_unit`` if no hyperlinks are
    present (with a warning), which is what happens for non-Wikipedia
    corpora in the canonical suite.
    """
    has_links = any(
        isinstance(d.metadata, dict) and d.metadata.get("hyperlinks")
        for d in documents
    )
    if not has_links:
        logger.warning(
            "hyperlink_based grouping requested but no CanonicalDocument "
            "has metadata['hyperlinks']; falling back to one_document_per_unit."
        )
        return _group_one_document_per_unit(documents)

    by_id = {d.doc_id: d for d in documents}
    parent: dict[str, str] = {d.doc_id: d.doc_id for d in documents}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for doc in documents:
        links = doc.metadata.get("hyperlinks") or []
        for target in links:
            if target in by_id:
                union(doc.doc_id, target)

    clusters: dict[str, list[CanonicalDocument]] = {}
    for doc in documents:
        clusters.setdefault(find(doc.doc_id), []).append(doc)

    units: list[_Unit] = []
    for members in clusters.values():
        units.extend(_group_adjacent_segments(members, target_tokens))

    # Renumber for stable unit_ids across clusters
    return [
        _Unit(
            unit_id=f"unit::hl{i:05d}",
            text=u.text,
            doc_ids=u.doc_ids,
            token_count=u.token_count,
            title_map=u.title_map,
        )
        for i, u in enumerate(units)
    ]


# ---------------------------------------------------------------------------
# Native embedder (lazy)
# ---------------------------------------------------------------------------

class _NativeBGEEmbedder:
    """Lazy loader for BGE-large-en-v1.5 used only in native mode.

    Model resolution order:
        1. ``model_path`` argument (from config extra.embedder_local_path).
        2. Environment variable ``LONGRAG_EMBEDDER_PATH``.
        3. HuggingFace hub name ``BAAI/bge-large-en-v1.5`` (requires cache
           or network access).

    BGE uses CLS pooling per the official model card.
    """

    def __init__(self, model_path: str | None = None) -> None:
        self._model_path = model_path
        self._model: Any = None
        self._tokenizer: Any = None
        self._device: str = "cpu"

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Native LongRAG mode requires `torch` and `transformers`. "
                "Install them (see workspace/longrag_dev/requirements.txt) "
                "or switch to controlled mode."
            ) from exc

        path = (
            self._model_path
            or os.environ.get("LONGRAG_EMBEDDER_PATH")
            or "BAAI/bge-large-en-v1.5"
        )
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._tokenizer = AutoTokenizer.from_pretrained(path)
        self._model = AutoModel.from_pretrained(path).to(self._device)
        self._model.eval()
        logger.info(
            "Loaded native BGE embedder from %s on %s", path, self._device
        )

    @property
    def name(self) -> str:
        return "BAAI/bge-large-en-v1.5"

    def encode_documents(self, texts: list[str]) -> list[list[float]]:
        self._load()
        import torch

        out: list[list[float]] = []
        batch_size = 8
        with torch.no_grad():
            for start in range(0, len(texts), batch_size):
                batch = texts[start : start + batch_size]
                enc = self._tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=8192,
                    return_tensors="pt",
                ).to(self._device)
                last_hidden = self._model(**enc).last_hidden_state
                cls = last_hidden[:, 0]  # CLS pooling per BGE model card
                cls = torch.nn.functional.normalize(cls, p=2, dim=-1)
                out.extend(cls.cpu().numpy().tolist())
        return out

    def encode_query(self, text: str) -> list[float]:
        return self.encode_documents([text])[0]


# ---------------------------------------------------------------------------
# Main adapter
# ---------------------------------------------------------------------------

class LongRAGBaseline(BaseBaseline):
    """LongRAG retrieval adapter (IR-only)."""

    name = "longrag"
    version = "0.1.0"
    status = BaselineStatus.EXECUTABLE
    supports_native = True
    supports_controlled = True
    is_llm_based = False

    def __init__(
        self,
        *,
        config: BaselineConfig,
        work_dir: Any,
        embedder: Embedder | None = None,
        llm: LLM | None = None,
    ) -> None:
        super().__init__(
            config=config,
            work_dir=work_dir,
            embedder=embedder,
            llm=llm,
        )
        self._target_unit_tokens: int = int(
            config.extra.get("target_unit_tokens", 4096)
        )
        self._k1: int = int(config.extra.get("k1", 100))
        self._grouping_policy: str = str(
            config.extra.get("grouping_policy", "one_document_per_unit")
        )
        self._native_embedder: _NativeBGEEmbedder | None = None

        # Cached index state
        self._unit_vectors: np.ndarray | None = None
        self._unit_records: list[dict[str, Any]] | None = None
        self._index_meta: dict[str, Any] | None = None

    # ---- embedder resolution ----

    def _get_embedder(self) -> Any:
        if self.config.mode == RunMode.CONTROLLED:
            if self.embedder is None:
                raise RuntimeError(
                    "Controlled mode requires an injected embedder."
                )
            return self.embedder
        if self._native_embedder is None:
            self._native_embedder = _NativeBGEEmbedder(
                model_path=self.config.extra.get("embedder_local_path"),
            )
        return self._native_embedder

    # ---- grouping ----

    def _group(self, documents: list[CanonicalDocument]) -> list[_Unit]:
        policy = self._grouping_policy
        if policy == "one_document_per_unit":
            return _group_one_document_per_unit(documents)
        if policy == "adjacent_document_segments":
            return _group_adjacent_segments(
                documents, target_tokens=self._target_unit_tokens
            )
        if policy == "hyperlink_based":
            return _group_hyperlink_based(
                documents, target_tokens=self._target_unit_tokens
            )
        raise ValueError(
            f"Unknown grouping_policy {policy!r}. Expected one of: "
            "'one_document_per_unit', 'adjacent_document_segments', "
            "'hyperlink_based'."
        )

    # ---- BaseBaseline: build_index ----

    def build_index(self, documents: list[CanonicalDocument]) -> None:
        if not documents:
            raise ValueError("Cannot build LongRAG index from empty corpus.")

        units = self._group(documents)
        if not units:
            raise ValueError("Grouping produced no units.")

        embedder = self._get_embedder()
        unit_texts = [u.text for u in units]
        vectors = embedder.encode_documents(unit_texts)
        if len(vectors) != len(units):
            raise RuntimeError(
                f"Embedder returned {len(vectors)} vectors for "
                f"{len(units)} units."
            )

        arr = np.asarray(vectors, dtype=np.float32)
        if arr.ndim != 2:
            raise RuntimeError(
                f"Embedder must return 2D vectors; got shape {arr.shape}."
            )
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        arr = arr / norms

        # Persist under work_dir
        vec_path = self.work_dir / "unit_vectors.npy"
        rec_path = self.work_dir / "unit_records.json"
        meta_path = self.work_dir / "index_meta.json"

        np.save(vec_path, arr)
        records: list[dict[str, Any]] = [
            {
                "unit_id": u.unit_id,
                "text": u.text,
                "doc_ids": list(u.doc_ids),
                "token_count": u.token_count,
                "title_map": dict(u.title_map),
            }
            for u in units
        ]
        rec_path.write_text(
            json.dumps(records, ensure_ascii=False), encoding="utf-8"
        )
        meta = {
            "name": self.name,
            "version": self.version,
            "mode": self.config.mode.value,
            "grouping_policy": self._grouping_policy,
            "target_unit_tokens": self._target_unit_tokens,
            "k1": self._k1,
            "embedder": getattr(embedder, "name", "unknown"),
            "dim": int(arr.shape[1]),
            "num_units": len(units),
            "num_docs": len(documents),
        }
        meta_path.write_text(
            json.dumps(meta, ensure_ascii=False), encoding="utf-8"
        )

        self._unit_vectors = arr
        self._unit_records = records
        self._index_meta = meta

    # ---- BaseBaseline: retrieve ----

    def _load_index(self) -> None:
        if self._unit_vectors is not None and self._unit_records is not None:
            return
        vec_path = self.work_dir / "unit_vectors.npy"
        rec_path = self.work_dir / "unit_records.json"
        if not vec_path.exists() or not rec_path.exists():
            raise RuntimeError(
                f"Index not found under {self.work_dir}. "
                "Call build_index() first."
            )
        self._unit_vectors = np.load(vec_path)
        self._unit_records = json.loads(rec_path.read_text(encoding="utf-8"))
        meta_path = self.work_dir / "index_meta.json"
        if meta_path.exists():
            self._index_meta = json.loads(
                meta_path.read_text(encoding="utf-8")
            )

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        self._load_index()
        if self._unit_vectors is None or self._unit_records is None:
            raise RuntimeError(
                "Internal error: index state missing after _load_index()."
            )

        k_final = top_k if top_k is not None else self.config.top_k
        if k_final <= 0:
            return []

        embedder = self._get_embedder()
        q = np.asarray(embedder.encode_query(query), dtype=np.float32)
        q_norm = float(np.linalg.norm(q))
        if q_norm > 0.0:
            q = q / q_norm

        # Step 1: score all units (cosine; unit vectors pre-normalized).
        scores = self._unit_vectors @ q
        n_units = int(scores.shape[0])
        if n_units == 0:
            return []
        k1 = min(self._k1, n_units)

        # Partial sort to get top-k1, then exact sort for determinism.
        if k1 < n_units:
            idx = np.argpartition(-scores, k1 - 1)[:k1]
        else:
            idx = np.arange(n_units)
        idx = idx[np.argsort(-scores[idx], kind="stable")]

        # Step 2: expand units -> documents (MaxP approximation).
        doc_best_score: dict[str, float] = {}
        doc_to_text: dict[str, str] = {}
        doc_to_unit: dict[str, str] = {}
        doc_to_title: dict[str, str] = {}
        for ui in idx:
            ui_int = int(ui)
            unit_score = float(scores[ui_int])
            rec = self._unit_records[ui_int]
            title_map = rec.get("title_map", {}) or {}
            for doc_id in rec["doc_ids"]:
                prev = doc_best_score.get(doc_id)
                if prev is None or unit_score > prev:
                    doc_best_score[doc_id] = unit_score
                # First unit that mentions the doc provides its text/title.
                if doc_id not in doc_to_text:
                    doc_to_text[doc_id] = rec["text"]
                    doc_to_unit[doc_id] = rec["unit_id"]
                    doc_to_title[doc_id] = title_map.get(doc_id, doc_id)

        # Step 3: rank documents (score desc, doc_id asc for determinism).
        ranked = sorted(
            doc_best_score.items(), key=lambda kv: (-kv[1], kv[0])
        )
        ranked = ranked[:k_final]

        # Step 4: build RetrievedChunk list with contiguous ranks.
        chunks: list[RetrievedChunk] = []
        for rank, (doc_id, score) in enumerate(ranked, start=1):
            chunks.append(
                RetrievedChunk(
                    doc_id=doc_id,
                    chunk_id=doc_id,
                    text=doc_to_text.get(doc_id, ""),
                    score=float(score),
                    rank=rank,
                    parent_doc_id=None,
                    metadata={
                        "unit_id": doc_to_unit.get(doc_id, ""),
                        "mode": self.config.mode.value,
                        "grouping_policy": self._grouping_policy,
                        "title": doc_to_title.get(doc_id, doc_id),
                    },
                )
            )
        return chunks

    # ---- lifecycle ----

    def cleanup(self) -> None:
        self._unit_vectors = None
        self._unit_records = None
        self._index_meta = None
        self._native_embedder = None