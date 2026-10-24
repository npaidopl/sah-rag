"""Multi-Meta-RAG baseline adapter (native + controlled).

Reference:
    Poliakov & Shvai, "Multi-Meta-RAG", ICTERI 2024 Posters. arXiv:2406.13213.
    https://github.com/mxpoliakov/Multi-Meta-RAG

Native mode (paper-exact / author-supported):
    - Metadata extraction: gpt-3.5-turbo-1106 (paper-native)
    - Embedder: bge-large-en-v1.5 (author-supported, local, free)
      * paper also tests voyage-02 (API) but chooses bge as the local
        alternative. bge-large-en-v1.5 is what makes reproduction
        feasible without paid APIs.
    - Reranker: bge-reranker-large (author-supported, optional; if
      weights missing, retrieval proceeds without rerank)
    - Vector store: in-memory numpy (semantically equivalent to the
      paper's Neo4j $in/$nin metadata filtering)
    - Chunking: 256 words / 32 overlap (word-based approximation of
      the paper's 256-token / 32-overlap LlamaIndex splitter)

Controlled mode:
    - Metadata extraction: deterministic word-boundary substring
      matching against corpus-derived source vocabulary
      (justified by companion paper arXiv:2607.03929:
       88.0% vs 80.9% set-exact accuracy on source filtering)
    - Embedder: injected (BGE-M3) by orchestrator
    - Reranker: not applied

is_llm_based=False is intentional. The base class uses this flag only
to decide whether controlled mode requires llm injection. Native mode
uses an LLM internally, but that is documented here and in README.
"""
from __future__ import annotations

import json
import os
import re
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


_DEFAULT_EMBEDDER_PATH  = r"D:\rag_work\models\embedders\bge-large-en-v1.5"
_DEFAULT_RERANKER_PATH  = r"D:\rag_work\models\rerankers\bge-reranker-large"
_DEFAULT_LLM_MODEL      = "gpt-3.5-turbo-1106"

_EXTRACTION_PROMPT = """Extract metadata from the following query.

Return a JSON object with two keys:
- "article source": the news source mentioned (e.g. "The Verge", "TechCrunch"), or null if none
- "publication date": the publication date mentioned (YYYY-MM-DD), or null if none

Query: {query}

Return only the JSON object, no explanation."""


class MultiMetaRAGBaseline(BaseBaseline):
    name = "multi_meta_rag"
    version = "0.2.0"
    status = BaselineStatus.EXECUTABLE
    supports_native = True
    supports_controlled = True
    is_llm_based = False  # see docstring: controlled mode is LLM-free

    def __init__(self, *, config, work_dir, embedder=None, llm=None):
        super().__init__(config=config, work_dir=work_dir, embedder=embedder, llm=llm)
        self._st_model  = None
        self._reranker  = None
        self._openai    = None
        if config.mode == RunMode.NATIVE:
            self._validate_native_prereqs()

    # ---------- path resolution ----------
    def _native_embedder_path(self) -> Path:
        p = self.config.extra.get("native_embedder_path")
        if p: return Path(p)
        return Path(os.environ.get("MULTI_META_RAG_EMBEDDER_PATH", _DEFAULT_EMBEDDER_PATH))

    def _native_reranker_path(self) -> Path:
        p = self.config.extra.get("native_reranker_path")
        if p: return Path(p)
        return Path(os.environ.get("MULTI_META_RAG_RERANKER_PATH", _DEFAULT_RERANKER_PATH))

    def _native_llm_model(self) -> str:
        return self.config.extra.get("native_llm_model", _DEFAULT_LLM_MODEL)

    # ---------- persistence ----------
    @property
    def _embeddings_path(self) -> Path: return self.work_dir / "embeddings.npy"
    @property
    def _docs_path(self) -> Path:       return self.work_dir / "docs.json"
    @property
    def _vocab_path(self) -> Path:      return self.work_dir / "source_vocab.json"

    # ---------- fail-fast ----------
    def _validate_native_prereqs(self) -> None:
        if self.embedder is None:
            p = self._native_embedder_path()
            if not p.exists():
                raise FileNotFoundError(
                    f"{self.name}: native mode requires embedder at {p}. "
                    f"Download bge-large-en-v1.5 there, or set "
                    f"MULTI_META_RAG_EMBEDDER_PATH env var."
                )
        if self.llm is None and not os.environ.get("OPENAI_API_KEY"):
            raise EnvironmentError(
                f"{self.name}: native mode requires OPENAI_API_KEY env var "
                f"(paper-exact extraction uses {self._native_llm_model()}). "
                f"Alternatively inject an LLM for testing."
            )

    # ---------- chunking ----------
    @staticmethod
    def _chunk_text(text: str, size_words: int, overlap_words: int) -> list[str]:
        words = text.split()
        if len(words) <= size_words:
            return [text] if text else []
        step = max(1, size_words - overlap_words)
        chunks = []
        for start in range(0, len(words), step):
            end = start + size_words
            cw = words[start:end]
            if not cw: break
            chunks.append(" ".join(cw))
            if end >= len(words): break
        return chunks

    # ---------- build ----------
    def build_index(self, documents: list[Any]) -> None:
        if not documents:
            raise ValueError(f"{self.name}: cannot build index from empty corpus")

        size_words    = int(self.config.extra.get("chunk_size_words", 256))
        overlap_words = int(self.config.extra.get("chunk_overlap_words", 32))

        all_texts: list[str] = []
        chunk_records: list[dict] = []
        vocab_set: set[str] = set()

        for d in documents:
            md = getattr(d, "metadata", None) or {}
            src = md.get("source") if isinstance(md, dict) else None
            if isinstance(src, str) and src.strip():
                vocab_set.add(src.strip())

            for i, ctext in enumerate(self._chunk_text(d.text_normalized, size_words, overlap_words)):
                all_texts.append(ctext)
                chunk_records.append({
                    "doc_id":   d.doc_id,
                    "chunk_id": f"{d.doc_id}::chunk_{i}",
                    "text":     ctext,
                    "source":   src,
                })

        embeddings = self._encode_documents(all_texts)
        np.save(self._embeddings_path, embeddings)
        self._docs_path.write_text(
            json.dumps(chunk_records, ensure_ascii=False, indent=2), encoding="utf-8")
        self._vocab_path.write_text(
            json.dumps(sorted(vocab_set), ensure_ascii=False, indent=2), encoding="utf-8")

    # ---------- encoding ----------
    def _encode_documents(self, texts: list[str]) -> np.ndarray:
        if self.embedder is not None:
            vecs = self.embedder.encode_documents(texts)
        else:
            if self._st_model is None:
                from sentence_transformers import SentenceTransformer
                self._st_model = SentenceTransformer(
                    str(self._native_embedder_path()), device="cpu")
            vecs = self._st_model.encode(
                texts, convert_to_numpy=True, show_progress_bar=False)
        arr = np.asarray(vecs, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return arr / norms

    def _encode_query(self, text: str) -> np.ndarray:
        if self.embedder is not None:
            vec = self.embedder.encode_query(text)
        else:
            if self._st_model is None:
                from sentence_transformers import SentenceTransformer
                self._st_model = SentenceTransformer(
                    str(self._native_embedder_path()), device="cpu")
            vec = self._st_model.encode(
                [text], convert_to_numpy=True, show_progress_bar=False)[0]
        v = np.asarray(vec, dtype=np.float32)
        n = np.linalg.norm(v)
        return v if n == 0 else v / n

    # ---------- extraction ----------
    def _extract_via_llm(self, query: str) -> dict:
        prompt = _EXTRACTION_PROMPT.format(query=query)
        if self.llm is not None:
            raw = self.llm.generate(prompt)
        else:
            if self._openai is None:
                from openai import OpenAI
                self._openai = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
            kwargs: dict = {
                "model": self._native_llm_model(),
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
            }
            if "gpt-3.5" in kwargs["model"] or "gpt-4" in kwargs["model"]:
                kwargs["response_format"] = {"type": "json_object"}
            resp = self._openai.chat.completions.create(**kwargs)
            raw = resp.choices[0].message.content
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return {}
        return parsed if isinstance(parsed, dict) else {}

    @staticmethod
    def _extract_via_substring(query: str, vocab: list[str]) -> list[str]:
        if not vocab: return []
        matched = []
        q_lower = query.lower()
        for src in vocab:
            if re.search(r"\b" + re.escape(src.lower()) + r"\b", q_lower):
                matched.append(src)
        return matched

    def _extract_sources(self, query: str, vocab: list[str]) -> list[str]:
        if self.config.mode == RunMode.NATIVE:
            raw = self._extract_via_llm(query)
            src = raw.get("article source")
            if src is None: return []
            if isinstance(src, list): return [s for s in src if s]
            if isinstance(src, str) and src.strip(): return [src.strip()]
            return []
        return self._extract_via_substring(query, vocab)

    # ---------- reranker ----------
    def _maybe_rerank(self, query: str, items: list[tuple[float, dict]]) -> list[tuple[float, dict]]:
        if self.config.mode != RunMode.NATIVE or not items:
            return items
        p = self._native_reranker_path()
        if not p.exists():
            return items
        if self._reranker is None:
            try:
                from FlagEmbedding import FlagReranker
                self._reranker = FlagReranker(str(p), use_fp16=False)
            except ImportError:
                return items
        pairs = [[query, r["text"]] for _, r in items]
        scores = self._reranker.compute_score(pairs)
        if isinstance(scores, (int, float)):
            scores = [scores]
        reranked = sorted(
            [(float(s), r) for s, (_, r) in zip(scores, items)],
            key=lambda x: -x[0],
        )
        return reranked

    # ---------- load ----------
    def _load_index(self):
        if not self._embeddings_path.exists():
            raise RuntimeError(f"{self.name}: index not built. Call build_index() first.")
        return (
            np.load(self._embeddings_path),
            json.loads(self._docs_path.read_text(encoding="utf-8")),
            json.loads(self._vocab_path.read_text(encoding="utf-8")),
        )

    # ---------- retrieve ----------
    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        k = top_k if top_k is not None else self.config.top_k
        if k <= 0: return []

        embeddings, records, vocab = self._load_index()
        if embeddings.size == 0 or not records: return []

        sources = self._extract_sources(query, vocab)
        if sources:
            src_set = set(sources)
            cand = [i for i, r in enumerate(records) if r.get("source") in src_set]
        else:
            cand = list(range(len(records)))
        if not cand: return []

        q = self._encode_query(query)
        scores = embeddings[cand] @ q
        order = np.argsort(-scores, kind="stable")

        # retrieve extra candidates to give reranker room to work
        n_pre = k * 4 if self.config.mode == RunMode.NATIVE else k
        items: list[tuple[float, dict]] = []
        for li in order[:n_pre]:
            gi = cand[int(li)]
            items.append((float(scores[int(li)]), records[gi]))

        if self.config.mode == RunMode.NATIVE:
            items = self._maybe_rerank(query, items)

        items = items[:k]
        return [
            RetrievedChunk(
                doc_id=r["doc_id"],
                chunk_id=r["chunk_id"],
                text=r["text"],
                score=s,
                rank=i,
                parent_doc_id=None,
                metadata={"source": r.get("source")},
            )
            for i, (s, r) in enumerate(items, start=1)
        ]