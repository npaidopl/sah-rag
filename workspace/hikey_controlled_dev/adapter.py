"""HiKEY baseline adapter (paper-derived controlled comparator).

Reference
---------
Joongmin Shin, Gyuho Shim, Jeongbae Park, Jaehyung Seo, Heuiseok Lim.
"HiKEY: Hierarchical Multimodal Retrieval for Open-Domain Document
Question Answering." ACL 2026.  arXiv:2605.29606.

Disposition
-----------
* Native track: NOT supported.  There is no public repository,
  no author-released configs, and no model weights.  The paper-
  reported values are exposed via the class attribute
  ``REPORTED_METRICS`` for downstream native comparisons.
* Controlled track: paper-derived text-only approximation that
  preserves the architecture-defining retrieval logic:
    - coarse-to-fine routing (doc routing, then section ranking)
    - MaxSim section scoring (max over intra-section units)
    - evidence assembly with governing-header context
  The LVLM reader and true multimodal units (tables/figures)
  are OUT OF SCOPE for v2 (IR-only).
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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


# ---------------------------------------------------------------------------
# Defaults / constants
# ---------------------------------------------------------------------------

DEFAULT_SECTION_SIZE = 2000   # characters per pseudo-section
DEFAULT_UNIT_SIZE = 500       # characters per retrieval unit
BM25_K1 = 1.5
BM25_B = 0.75
TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


# ---------------------------------------------------------------------------
# Internal data structures
# ---------------------------------------------------------------------------

@dataclass
class _Unit:
    unit_id: str
    text: str
    embedding: list[float] | None = None


@dataclass
class _Section:
    section_id: str
    doc_id: str
    title: str
    text: str
    level: int
    parent_section_id: str | None
    units: list[_Unit] = field(default_factory=list)
    embedding: list[float] | None = None


@dataclass
class _Doc:
    doc_id: str
    title: str
    text: str
    sections: list[_Section] = field(default_factory=list)
    embedding: list[float] | None = None
    tokens: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall((text or "").lower())


def _chunk_text(text: str, size: int) -> list[str]:
    text = text or ""
    if not text.strip():
        return []
    if len(text) <= size:
        return [text]
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            space = text.rfind(" ", start, end)
            if space > start + size // 2:
                end = space
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        start = end if end > start else start + size
    return chunks


def _extract_sections(doc: CanonicalDocument) -> list[dict[str, Any]]:
    """Extract section structure.

    Priority:
      1. ``metadata['sections']`` (list of {section_id?, title, text,
         level?, parent_id?})
      2. Fallback: split ``text_normalized``/``text_raw`` by size.
    """
    meta = doc.metadata or {}
    sections = meta.get("sections")
    if isinstance(sections, list) and sections:
        out: list[dict[str, Any]] = []
        for i, sec in enumerate(sections):
            if not isinstance(sec, dict):
                continue
            title = str(sec.get("title", "") or "")
            text = str(sec.get("text", "") or "")
            level = int(sec.get("level", 0) or 0)
            parent = sec.get("parent_id")
            if not (text.strip() or title.strip()):
                continue
            out.append({
                "section_id": str(sec.get("section_id", f"sec_{i}")),
                "title": title,
                "text": text,
                "level": level,
                "parent_id": str(parent) if parent else None,
            })
        if out:
            return out

    source = doc.text_normalized or doc.text_raw or ""
    chunks = _chunk_text(source, DEFAULT_SECTION_SIZE)
    return [
        {
            "section_id": f"sec_{i}",
            "title": f"Section {i + 1}",
            "text": chunk,
            "level": 0,
            "parent_id": None,
        }
        for i, chunk in enumerate(chunks)
    ]


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

class HiKEYBaseline(BaseBaseline):
    """HiKEY baseline — reported-only for native, paper-derived for controlled."""

    name = "hikey_controlled"
    version = "0.2.0"
    status = BaselineStatus.EXECUTABLE
    supports_native = False
    supports_controlled = True
    is_llm_based = False

    # ------------------------------------------------------------------
    # Paper-reported native metrics (fill in from the paper when ready).
    # Structure: dataset_id -> {metric_name: value}
    # ------------------------------------------------------------------
    REPORTED_METRICS: dict[str, dict[str, float]] = {
        # "long_bench_v2": {"recall@5": ..., "ndcg@10": ...},
    }

    # ------------------------------------------------------------------
    # Init
    # ------------------------------------------------------------------

    def __init__(
        self,
        *,
        config: BaselineConfig,
        work_dir: Path,
        embedder: Embedder | None = None,
        llm: LLM | None = None,
    ) -> None:
        super().__init__(
            config=config, work_dir=work_dir, embedder=embedder, llm=llm
        )
        extra = self.config.extra or {}
        self.doc_route_top_n: int = int(
            extra.get("doc_route_top_n", max(self.config.top_k * 3, 20))
        )
        self.section_size: int = int(extra.get("section_size", DEFAULT_SECTION_SIZE))
        self.unit_size: int = int(extra.get("unit_size", DEFAULT_UNIT_SIZE))
        self.dense_weight: float = float(extra.get("dense_weight", 0.7))
        self.lexical_weight: float = float(extra.get("lexical_weight", 0.3))

        # In-memory index (persisted manifest under work_dir)
        self._docs: list[_Doc] = []
        self._all_tokens: list[list[str]] = []
        self._doc_freq: dict[str, int] = {}
        self._avg_doc_len: float = 0.0
        self._n_docs: int = 0

    # ------------------------------------------------------------------
    # build_index
    # ------------------------------------------------------------------

    def build_index(self, documents: list[CanonicalDocument]) -> None:
        if not documents:
            self._docs = []
            self._all_tokens = []
            self._doc_freq = {}
            self._avg_doc_len = 0.0
            self._n_docs = 0
            self._persist_manifest()
            return

        docs: list[_Doc] = []
        for doc in documents:
            sections_data = _extract_sections(doc)
            sections: list[_Section] = []
            for s in sections_data:
                unit_texts = _chunk_text(s["text"], self.unit_size)
                if not unit_texts and s["title"]:
                    unit_texts = [s["title"]]
                units = [
                    _Unit(unit_id=f"{s['section_id']}_u{i}", text=t)
                    for i, t in enumerate(unit_texts)
                ]
                sections.append(_Section(
                    section_id=s["section_id"],
                    doc_id=doc.doc_id,
                    title=s["title"],
                    text=s["text"],
                    level=s["level"],
                    parent_section_id=s["parent_id"],
                    units=units,
                ))
            full_text = doc.text_normalized or doc.text_raw or ""
            docs.append(_Doc(
                doc_id=doc.doc_id,
                title=doc.title or "",
                text=full_text,
                sections=sections,
            ))

        # --- Embeddings (controlled mode guarantees self.embedder) ---
        if self.embedder is not None:
            all_texts: list[str] = []
            for d in docs:
                all_texts.append(((d.title + "\n" + d.text) or "")[:8000])
                for sec in d.sections:
                    all_texts.append(((sec.title + "\n" + sec.text) or "")[:4000])
                    for unit in sec.units:
                        all_texts.append(unit.text or "")
            embeddings = self.embedder.encode_documents(all_texts)
            idx = 0
            for d in docs:
                d.embedding = embeddings[idx]; idx += 1
                for sec in d.sections:
                    sec.embedding = embeddings[idx]; idx += 1
                    for unit in sec.units:
                        unit.embedding = embeddings[idx]; idx += 1

        # --- BM25 statistics ---
        doc_freq: Counter[str] = Counter()
        all_tokens: list[list[str]] = []
        for d in docs:
            toks = _tokenize(d.text)
            d.tokens = toks
            all_tokens.append(toks)
            for t in set(toks):
                doc_freq[t] += 1

        self._docs = docs
        self._all_tokens = all_tokens
        self._doc_freq = dict(doc_freq)
        self._n_docs = len(docs)
        self._avg_doc_len = (
            sum(len(t) for t in all_tokens) / self._n_docs
            if self._n_docs else 0.0
        )
        self._persist_manifest()

    # ------------------------------------------------------------------
    # Scoring helpers
    # ------------------------------------------------------------------

    def _bm25(self, q_tokens: list[str], doc_idx: int) -> float:
        tokens = self._all_tokens[doc_idx]
        if not tokens or not q_tokens:
            return 0.0
        tf = Counter(tokens)
        dl = len(tokens)
        score = 0.0
        for qt in set(q_tokens):
            df = self._doc_freq.get(qt, 0)
            if df == 0:
                continue
            idf = math.log(1.0 + (self._n_docs - df + 0.5) / (df + 0.5))
            f = tf.get(qt, 0)
            denom = f + BM25_K1 * (
                1.0 - BM25_B + BM25_B * dl / max(self._avg_doc_len, 1e-9)
            )
            if denom > 0:
                score += idf * (f * (BM25_K1 + 1.0)) / denom
        return score

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        if not a or not b:
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        if na == 0.0 or nb == 0.0:
            return 0.0
        return dot / (na * nb)

    def _persist_manifest(self) -> None:
        manifest = {
            "adapter": self.name,
            "version": self.version,
            "n_docs": self._n_docs,
            "n_sections": sum(len(d.sections) for d in self._docs),
            "avg_doc_len": self._avg_doc_len,
            "params": {
                "doc_route_top_n": self.doc_route_top_n,
                "section_size": self.section_size,
                "unit_size": self.unit_size,
                "dense_weight": self.dense_weight,
                "lexical_weight": self.lexical_weight,
            },
        }
        (self.work_dir / "index_manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )

    # ------------------------------------------------------------------
    # retrieve
    # ------------------------------------------------------------------

    def retrieve(
        self, query: str, top_k: int | None = None
    ) -> list[RetrievedChunk]:
        if not self._docs:
            return []
        k = top_k if top_k is not None else self.config.top_k
        if k <= 0:
            return []

        q_tokens = _tokenize(query)
        q_emb = (
            self.embedder.encode_query(query)
            if self.embedder is not None else None
        )

        # ---- Stage 1: global document routing ----
        doc_scores: list[tuple[int, float]] = []
        for di, d in enumerate(self._docs):
            lex = self._bm25(q_tokens, di)
            if lex > 0 and self._avg_doc_len > 0:
                lex = lex / (lex + 5.0)  # squash to [0, 1)
            den = 0.0
            if q_emb is not None and d.embedding is not None:
                den = self._cosine(q_emb, d.embedding)
            score = self.dense_weight * den + self.lexical_weight * lex
            doc_scores.append((di, score))
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        routed = doc_scores[: self.doc_route_top_n]

        # ---- Stage 2: section-level MaxSim ----
        section_scores: list[tuple[int, int, float]] = []
        for di, doc_score in routed:
            d = self._docs[di]
            for si, sec in enumerate(d.sections):
                if q_emb is not None and sec.units:
                    unit_sims = [
                        self._cosine(q_emb, u.embedding)
                        for u in sec.units
                        if u.embedding is not None
                    ]
                    maxsim = max(unit_sims) if unit_sims else 0.0
                else:
                    maxsim = 0.0
                    if sec.text and q_tokens:
                        stoks = _tokenize(sec.text)
                        if stoks:
                            overlap = (
                                len(set(q_tokens) & set(stoks))
                                / max(len(set(q_tokens)), 1)
                            )
                            maxsim = overlap
                final = 0.7 * maxsim + 0.3 * doc_score
                section_scores.append((di, si, final))

        section_scores.sort(key=lambda x: x[2], reverse=True)
        selected = section_scores[:k]

        # ---- Evidence assembly ----
        results: list[RetrievedChunk] = []
        for rank, (di, si, score) in enumerate(selected, start=1):
            d = self._docs[di]
            sec = d.sections[si]

            # Governing headers (parent title, then own title)
            prefix_parts: list[str] = []
            if sec.parent_section_id:
                for parent in d.sections:
                    if (
                        parent.section_id == sec.parent_section_id
                        and parent.title
                    ):
                        prefix_parts.append(f"[{parent.title}]")
                        break
            if sec.title:
                prefix_parts.append(f"[{sec.title}]")
            header = " ".join(prefix_parts)

            text = (header + "\n" + sec.text).strip() if header else sec.text
            if not text.strip() and sec.units:
                text = "\n".join(u.text for u in sec.units)
            if not text.strip():
                text = sec.title or f"section_{si}"

            results.append(RetrievedChunk(
                doc_id=d.doc_id,
                chunk_id=sec.section_id,
                text=text,
                score=float(score),
                rank=rank,
                parent_doc_id=None,
                metadata={
                    "section_title": sec.title,
                    "section_level": sec.level,
                    "approximation": "hiKEY_text_only_approximation",
                },
            ))
        return results