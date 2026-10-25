"""LightRAG baseline adapter for SAH-RAG v2.

Implements ``BaseBaseline`` for the LightRAG architecture
(arXiv:2410.05779, MIT, HKUDS/LightRAG).

Two execution paths
-------------------
Full vendored path (GPU server)
    When the vendored LightRAG is importable AND components are
    available, run the official pipeline: native chunking (1200/100),
    LLM-based entity/relation extraction, native prompts, dual-level
    retrieval. Native mode uses HuggingFace binding; controlled mode
    uses injected Embedder + LLM. Generation is disabled (IR-only).

Fallback path (offline / CI)
    When components are unavailable, run a lightweight pipeline that
    preserves LightRAG architectural ideas (fixed chunking,
    co-occurrence graph, dual-level retrieval fusion) without
    requiring models. This keeps the contract test offline and
    CPU-only.

Tags
----
native       ``AUTHOR_SUPPORTED_NATIVE``
controlled   ``CONTROLLED_ADAPTATION``

Never mislabel the controlled or fallback path as paper-exact native.
"""
from __future__ import annotations

import hashlib
import os
import json
import logging
import math
import re
import sys
from collections import Counter, defaultdict
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

logger = logging.getLogger(__name__)


# LightRAG native defaults (author-supported native).
NATIVE_CHUNK_SIZE = 1200
NATIVE_CHUNK_OVERLAP = 100


class LightRAGBaseline(BaseBaseline):
    """LightRAG baseline adapter (native + controlled)."""

    name = "lightrag"
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
        embedder: Embedder | None = None,
        llm: LLM | None = None,
    ) -> None:
        super().__init__(
            config=config,
            work_dir=work_dir,
            embedder=embedder,
            llm=llm,
        )
        self.chunk_size = int(
            config.extra.get("chunk_size", NATIVE_CHUNK_SIZE)
        )
        self.chunk_overlap = int(
            config.extra.get("chunk_overlap", NATIVE_CHUNK_OVERLAP)
        )
        self._chunks_path = self.work_dir / "chunks.jsonl"
        self._graph_path = self.work_dir / "graph.json"
        self._index_path = self.work_dir / "index.json"

        self._chunks: list[dict[str, Any]] = []
        self._graph: dict[str, dict[str, float]] = {}
        self._chunk_terms: list[Counter] = []
        self._chunk_embeddings: list[list[float]] | None = None

        self._vendored_path = self._find_vendored_path()
        self._native_components_loaded = False
        self._native_llm_name: str | None = None
        self._native_llm_func: Any | None = None
        self._native_embed_func: Any | None = None
        self._native_embed_tokenizer: Any | None = None
        self._native_embed_model: Any | None = None
        self._full_rag: Any | None = None
        if config.mode == RunMode.NATIVE:
            self._try_load_native_components()

    # ------------------------------------------------------------------
    # Vendored / native component discovery
    # ------------------------------------------------------------------
    @staticmethod
    def _find_vendored_path() -> Path | None:
        here = Path(__file__).resolve()
        for parent in here.parents:
            for sub in (
                ("external", "baselines", "lightrag"),
                ("external", "lightrag"),
                ("vendor", "lightrag"),
                ("third_party", "lightrag"),
            ):
                candidate = parent.joinpath(*sub)
                if (
                    candidate.is_dir()
                    and (candidate / "lightrag").is_dir()
                    and (candidate / "lightrag" / "__init__.py").is_file()
                ):
                    return candidate
        return None

    @staticmethod
    def _resolve_snapshot(model_name: str) -> str | None:
        """Find a complete local snapshot for a model (offline-safe).

        HF hub has two cache layouts:
          <HF_HUB_CACHE>/models--<org>--<name>/snapshots/<hash>/ (modern)
          <HF_HOME>/models--<org>--<name>/snapshots/<hash>/      (legacy)

        When HF_HUB_CACHE points at a partial snapshot, fall back to the
        other layout. Returns the path to a directory that has
        config.json + weight files, or None if nothing complete found.
        """
        import os as _os
        from pathlib import Path as _Path

        safe = model_name.replace("/", "--")
        hf_home = _os.environ.get("HF_HOME", "")
        hf_hub_cache = _os.environ.get("HF_HUB_CACHE", "")
        candidates = []
        if hf_hub_cache:
            candidates.append(
                _Path(hf_hub_cache) / f"models--{safe}" / "snapshots"
            )
        if hf_home:
            candidates.append(
                _Path(hf_home) / "hub" / f"models--{safe}" / "snapshots"
            )
            candidates.append(
                _Path(hf_home) / f"models--{safe}" / "snapshots"
            )
        for snapdir in candidates:
            if not snapdir.is_dir():
                continue
            for snap in snapdir.iterdir():
                if not snap.is_dir():
                    continue
                if not (snap / "config.json").exists():
                    continue
                has_weights = (
                    (snap / "pytorch_model.bin").exists()
                    or (snap / "model.safetensors").exists()
                    or (snap / "model.safetensors.index.json").exists()
                    or (snap / "pytorch_model.bin.index.json").exists()
                    or any(snap.glob("model-*-of-*.safetensors"))
                    or any(snap.glob("pytorch_model-*-of-*.bin"))
                )
                if has_weights:
                    return str(snap)
        return None

    def _try_load_native_components(self) -> None:
        """Load HF binding for native mode (no-op if not configured).

        Based on lightrag/llm/hf.py source inspection:

          - hf_model_complete(prompt, ...) gets the model name from
            kwargs["hashing_kv"].global_config["llm_model_name"], which
            LightRAG populates from the llm_model_name argument to
            LightRAG(...). So we pass the raw function and let LightRAG
            inject the model name.

          - hf_embed is an EmbeddingFunc whose .func coroutine expects
            (texts, tokenizer, embed_model, context=..., ...). We
            pre-load tokenizer + model once and wrap in a custom
            EmbeddingFunc whose .func closes over them.
        """
        if self._vendored_path is None:
            return
        hf_llm = self.config.extra.get("hf_llm")
        hf_embed_model = self.config.extra.get("hf_embed")
        hf_embed_dim = int(self.config.extra.get("hf_embed_dim", 1024))
        if not hf_llm or not hf_embed_model:
            return
        # Force offline cache use (network unavailable in acceptance runs,
        # and HF/XET retries cause 3+ minute stalls before falling back).
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
        try:
            if str(self._vendored_path) not in sys.path:
                sys.path.insert(0, str(self._vendored_path))
            from lightrag.llm.hf import (
                hf_embed as hf_embed_fn,
                hf_model_complete,
            )
            from lightrag.utils import EmbeddingFunc
            from transformers import AutoModel, AutoTokenizer

            # LLM: resolve snapshot; set _native_llm_name to snapshot
            # path if found (so initialize_hf_model loads the complete
            # copy), else fall back to model id.
            self._native_llm_func = hf_model_complete
            llm_snap = self._resolve_snapshot(hf_llm)
            if llm_snap:
                self._native_llm_name = llm_snap
                logger.info("LLM snapshot resolved: %s", llm_snap)
            else:
                self._native_llm_name = hf_llm
                logger.info("LLM snapshot unresolved; using id: %s", hf_llm)

            # Embeddings: pre-load tokenizer + model, wrap in EmbeddingFunc.
            # Prefer complete snapshot path when available.
            embed_snap = self._resolve_snapshot(hf_embed_model)
            embed_load_path = embed_snap or hf_embed_model
            if embed_snap:
                logger.info("Embed snapshot resolved: %s", embed_snap)
            else:
                logger.info("Embed snapshot unresolved; using id: %s", hf_embed_model)
            embed_tokenizer = AutoTokenizer.from_pretrained(
                embed_load_path, trust_remote_code=True, local_files_only=True
            )
            embed_model = AutoModel.from_pretrained(
                embed_load_path, trust_remote_code=True, local_files_only=True
            )
            self._native_embed_tokenizer = embed_tokenizer
            self._native_embed_model = embed_model

            async def _embed(texts, context="document"):
                return await hf_embed_fn.func(
                    texts, embed_tokenizer, embed_model,
                    context=context,
                )

            self._native_embed_func = EmbeddingFunc(
                embedding_dim=hf_embed_dim,
                max_token_size=8192,
                func=_embed,
                model_name=hf_embed_model,
            )
            self._native_components_loaded = True
            logger.info(
                "LightRAG native HF binding loaded "
                "(llm=%s, embed=%s, dim=%d)",
                hf_llm, hf_embed_model, hf_embed_dim,
            )
        except Exception as exc:
            logger.warning("Native HF binding unavailable: %s", exc)
            import traceback
            traceback.print_exc()

    def _should_use_full(self) -> bool:
        if self.config.extra.get("disable_full_pipeline", False):
            return False
        if self._vendored_path is None:
            return False
        if self.config.mode == RunMode.CONTROLLED:
            return self.embedder is not None and self.llm is not None
        return self._native_components_loaded

    # ------------------------------------------------------------------
    # build_index
    # ------------------------------------------------------------------
    def build_index(self, documents: list[CanonicalDocument]) -> None:
        strict = bool(self.config.extra.get("strict_full_path", False))
        if strict and not self._should_use_full():
            raise RuntimeError(
                f"{self.name}: strict_full_path=true but full pipeline is "
                f"not available (vendored={self._vendored_path}, "
                f"components={self._native_components_loaded or (self.embedder is not None and self.llm is not None)}). "
                f"Either provide components or set strict_full_path: false."
            )
        if self._should_use_full():
            try:
                self._build_index_full(documents)
                return
            except NotImplementedError:
                raise
            except Exception as exc:
                logger.warning(
                    "Full LightRAG pipeline failed, using fallback: %s",
                    exc,
                )
        self._build_index_fallback(documents)

    def _build_index_fallback(
        self, documents: list[CanonicalDocument]
    ) -> None:
        chunks: list[dict[str, Any]] = []
        for doc in documents:
            text = doc.text_normalized or doc.text_raw
            if not text:
                continue
            for cidx, chunk_text in enumerate(
                _fixed_chunk(text, self.chunk_size, self.chunk_overlap)
            ):
                chunks.append(
                    {
                        "doc_id": doc.doc_id,
                        "chunk_id": f"{doc.doc_id}::chunk::{cidx}",
                        "text": chunk_text,
                        "position": cidx,
                    }
                )
        if not chunks:
            self._write_empty()
            return
        graph = _build_cooccurrence_graph(chunks)
        chunk_terms = [Counter(_tokenize(c["text"])) for c in chunks]
        texts = [c["text"] for c in chunks]
        if self.embedder is not None:
            chunk_embeddings = self.embedder.encode_documents(texts)
        else:
            chunk_embeddings = [_fallback_embed(t) for t in texts]
        self._chunks = chunks
        self._graph = graph
        self._chunk_terms = chunk_terms
        self._chunk_embeddings = chunk_embeddings
        self._write_artifacts()

    def _build_index_full(
        self, documents: list[CanonicalDocument]
    ) -> None:
        """Full vendored LightRAG pipeline (real execution).

        Native mode:     uses HF binding (hf_model_complete + hf_embed).
        Controlled mode: wraps injected Embedder + LLM into LightRAG's
                         EmbeddingFunc / llm_model_func contracts.
        """
        if self._vendored_path is None:
            raise RuntimeError(
                "Vendored LightRAG not found at external/baselines/lightrag/."
            )
        if self.config.mode == RunMode.CONTROLLED:
            if self.embedder is None or self.llm is None:
                raise RuntimeError("Controlled mode requires embedder + llm.")
        else:
            if not self._native_components_loaded:
                raise RuntimeError(
                    "Native mode requires hf_llm + hf_embed in config.extra."
                )
        self._run_async(self._build_index_full_async(documents))

    async def _build_index_full_async(
        self, documents: list[CanonicalDocument]
    ) -> None:
        import numpy as np
        if str(self._vendored_path) not in sys.path:
            sys.path.insert(0, str(self._vendored_path))
        from lightrag import LightRAG
        from lightrag.utils import EmbeddingFunc

        if self.config.mode == RunMode.CONTROLLED:
            sample = self.embedder.encode_documents(["x"])
            embed_dim = len(sample[0]) if sample else 1024
            embedder_ref = self.embedder
            llm_ref = self.llm

            async def _embed(texts):
                return np.array(
                    embedder_ref.encode_documents(texts), dtype=np.float32
                )

            async def _llm(prompt, **kwargs):
                return llm_ref.generate(prompt, **kwargs)

            embedding_func = EmbeddingFunc(
                embedding_dim=embed_dim, max_token_size=8192, func=_embed,
            )
            llm_model_func = _llm
        else:
            # _native_embed_func is already an EmbeddingFunc instance
            embedding_func = self._native_embed_func
            llm_model_func = self._native_llm_func

        lightrag_dir = self.work_dir / "lightrag"
        lightrag_dir.mkdir(parents=True, exist_ok=True)
        if self.config.mode == RunMode.CONTROLLED:
            llm_name = getattr(self.llm, "name", "controlled_llm")
        else:
            llm_name = self._native_llm_name or self.config.extra.get(
                "hf_llm", "hf_llm"
            )
        try:
            rag = LightRAG(
                working_dir=str(lightrag_dir),
                llm_model_func=llm_model_func,
                embedding_func=embedding_func,
                chunk_token_size=self.chunk_size,
                chunk_overlap_token_size=self.chunk_overlap,
                llm_model_name=llm_name,
            )
        except TypeError:
            logger.warning(
                "LightRAG rejected llm_model_name kwarg; retrying without it"
            )
            rag = LightRAG(
                working_dir=str(lightrag_dir),
                llm_model_func=llm_model_func,
                embedding_func=embedding_func,
                chunk_token_size=self.chunk_size,
                chunk_overlap_token_size=self.chunk_overlap,
            )
        try:
            await rag.initialize_storages()
        except AttributeError:
            logger.warning("LightRAG.initialize_storages not available")
        except Exception as exc:
            logger.warning("initialize_storages raised: %s", exc)

        for doc in documents:
            text = doc.text_normalized or doc.text_raw
            if text:
                await rag.ainsert(text, ids=doc.doc_id)

        try:
            if hasattr(rag, "finalize_storages"):
                await rag.finalize_storages()
        except AttributeError:
            logger.warning("LightRAG.finalize_storages not available")
        except Exception as exc:
            logger.warning("finalize_storages raised: %s", exc)

        self._full_rag = rag

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        k = top_k if top_k is not None else self.config.top_k
        strict = bool(self.config.extra.get("strict_full_path", False))
        if strict and not self._should_use_full():
            raise RuntimeError(
                f"{self.name}: strict_full_path=true but full pipeline is "
                f"not available (vendored={self._vendored_path}, "
                f"components={self._native_components_loaded or (self.embedder is not None and self.llm is not None)}). "
                f"Either provide components or set strict_full_path: false."
            )
        if k <= 0 or not self._chunks:
            return []
        if self._should_use_full():
            try:
                return self._retrieve_full(query, k)
            except NotImplementedError:
                raise
            except Exception as exc:
                logger.warning(
                    "Full retrieval failed, using fallback: %s", exc
                )
        return self._retrieve_fallback(query, k)

    def _retrieve_fallback(
        self, query: str, k: int
    ) -> list[RetrievedChunk]:
        if (
            self._chunk_embeddings is not None
            and self.embedder is not None
        ):
            q_emb = self.embedder.encode_query(query)
            sims = [_cosine(q_emb, e) for e in self._chunk_embeddings]
        else:
            q_terms = Counter(_tokenize(query))
            sims = [
                _term_overlap_score(q_terms, terms)
                for terms in self._chunk_terms
            ]
        keywords = _extract_keywords(query)
        graph_boost = [0.0] * len(self._chunks)
        for kw in keywords:
            neighbors = self._graph.get(kw)
            if not neighbors:
                continue
            for idx, terms in enumerate(self._chunk_terms):
                if kw in terms:
                    graph_boost[idx] += 0.5
                    for neigh, weight in neighbors.items():
                        if neigh in terms:
                            graph_boost[idx] += weight
        fused = [
            sims[i] + 0.3 * graph_boost[i]
            for i in range(len(self._chunks))
        ]
        ranked = sorted(
            range(len(fused)),
            key=lambda i: fused[i],
            reverse=True,
        )[:k]
        results: list[RetrievedChunk] = []
        for rank, idx in enumerate(ranked, start=1):
            chunk = self._chunks[idx]
            results.append(
                RetrievedChunk(
                    doc_id=chunk["doc_id"],
                    chunk_id=chunk["chunk_id"],
                    text=chunk["text"],
                    score=float(fused[idx]),
                    rank=rank,
                    parent_doc_id=chunk["doc_id"],
                    metadata={"position": chunk.get("position", 0)},
                )
            )
        return results

    def _retrieve_full(
        self, query: str, k: int
    ) -> list[RetrievedChunk]:
        """Full vendored LightRAG retrieval (real execution)."""
        if self._full_rag is None:
            raise RuntimeError(
                "Full LightRAG not built. Call build_index() first."
            )
        return self._run_async(self._retrieve_full_async(query, k))

    async def _retrieve_full_async(
        self, query: str, k: int
    ) -> list[RetrievedChunk]:
        if self._full_rag is None:
            raise RuntimeError("Full LightRAG not built")
        try:
            from lightrag import QueryParam
        except ImportError:
            from lightrag.base import QueryParam
        param = QueryParam(
            mode="hybrid",
            top_k=k,
            chunk_top_k=k,
            only_need_context=True,
        )
        result_str = await self._full_rag.aquery(query, param=param)
        return self._parse_context_to_chunks(result_str, k)

    def _parse_context_to_chunks(self, context_str, k):
        """Parse LightRAG formatted context into RetrievedChunks.

        LightRAG's only_need_context=True output has multiple possible
        formats depending on mode:
          - naive:    "Document Chunks: {json-list-of-chunks-with-reference_id}"
          - local/global/hybrid:
                "Knowledge Graph Data (Entity): {json}\n...\nDocument Chunks: {...}"

        Chunk text may be escaped (JSON), truncated, or re-formatted.
        Full-text hash lookup is therefore unreliable.

        Four strategies, best first:
          1. chunk_id substring match — search "doc-x-chunk-N" tokens.
             Most robust: chunk_id is a stable short identifier.
          2. Partial text match — first 60 chars of each stored chunk.
             Tolerates trailing truncation or escaping of long chunks.
          3. Full text match — legacy exact-hash approach.
          4. Delimiter split — old fallback, keeps unmatched pieces.

        Ranking is native (order of first appearance in context). Scores
        are synthetic proxies (k - rank + 1). No fabricated doc_ids:
        unmatched chunks are marked matched=False with doc_id="unknown::N".
        """
        if not isinstance(context_str, str) or not context_str.strip():
            return []
        stored = self._load_stored_chunks()
        if not stored:
            return []

        def _build_results(matched_infos):
            results = []
            seen = set()
            unique = []
            for pos, info in sorted(matched_infos, key=lambda x: x[0]):
                cid = info["chunk_id"]
                if cid in seen:
                    continue
                seen.add(cid)
                unique.append((pos, info))
            for rank, (pos, info) in enumerate(unique[:k], start=1):
                results.append(
                    RetrievedChunk(
                        doc_id=info["doc_id"],
                        chunk_id=info["chunk_id"],
                        text=info["text"],
                        score=float(k - rank + 1),
                        rank=rank,
                        parent_doc_id=info["doc_id"],
                        metadata={
                            "source": "full_pipeline",
                            "matched": True,
                            "context_pos": pos,
                        },
                    )
                )
            return results

        # Strategy 1: chunk_id substring
        matched = []
        for info in stored.values():
            cid = info["chunk_id"]
            if not cid:
                continue
            pos = context_str.find(cid)
            if pos >= 0:
                matched.append((pos, info))
        if matched:
            return _build_results(matched)

        # Strategy 2: partial text (first 60 chars, after escape normalization)
        matched = []
        for info in stored.values():
            text = info.get("text", "")
            if len(text) < 20:
                continue
            needle = text[:60]
            # unescape common JSON escapes in needle if context is JSON
            pos = context_str.find(needle)
            if pos < 0:
                # try with JSON escapes on quotes/backslashes
                alt = needle.replace('\\', '\\\\').replace('"', '\\"')
                pos = context_str.find(alt)
            if pos >= 0:
                matched.append((pos, info))
        if matched:
            return _build_results(matched)

        # Strategy 3: full text (legacy)
        matched = []
        for info in stored.values():
            text = info.get("text", "")
            if not text or len(text) < 10:
                continue
            pos = context_str.find(text)
            if pos >= 0:
                matched.append((pos, info))
        if matched:
            return _build_results(matched)

        # Strategy 4: delimiter split with unmatched fallback
        pieces = []
        for delimiter in ("--New Chunk--", "-----"):
            if delimiter in context_str:
                pieces = [
                    p.strip()
                    for p in context_str.split(delimiter)
                    if p.strip()
                ]
                break
        if not pieces:
            pieces = [context_str.strip()]
        results = []
        for rank, piece in enumerate(pieces[:k], start=1):
            content_hash = hashlib.md5(piece.encode("utf-8")).hexdigest()
            info = stored.get(content_hash)
            if info:
                results.append(
                    RetrievedChunk(
                        doc_id=info["doc_id"],
                        chunk_id=info["chunk_id"],
                        text=info["text"] or piece,
                        score=float(k - rank + 1),
                        rank=rank,
                        parent_doc_id=info["doc_id"],
                        metadata={"source": "full_pipeline", "matched": True},
                    )
                )
            else:
                results.append(
                    RetrievedChunk(
                        doc_id="unknown::" + str(rank),
                        chunk_id="retrieved::" + str(rank),
                        text=piece,
                        score=float(k - rank + 1),
                        rank=rank,
                        parent_doc_id=None,
                        metadata={"source": "full_pipeline", "matched": False},
                    )
                )
        return results

    def _load_stored_chunks(self):
        """Load content_hash -> chunk info map from LightRAG KV store."""
        kv_path = self.work_dir / "lightrag" / "kv_store_text_chunks.json"
        if not kv_path.exists():
            return {}
        try:
            with kv_path.open(encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to load stored chunks: %s", exc)
            return {}
        mapping = {}
        for chunk_id, info in data.items():
            if not isinstance(info, dict):
                continue
            chunk_content = info.get("content", "")
            if not chunk_content:
                continue
            h = hashlib.md5(chunk_content.encode("utf-8")).hexdigest()
            mapping[h] = {
                "doc_id": info.get("full_doc_id", "unknown"),
                "chunk_id": chunk_id,
                "text": chunk_content,
            }
        return mapping

    def _run_async(self, coro):
        """Run an async coroutine from sync code."""
        import asyncio
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            return ex.submit(asyncio.run, coro).result()

    def _write_artifacts(self) -> None:
        with self._chunks_path.open("w", encoding="utf-8") as fh:
            for chunk in self._chunks:
                fh.write(json.dumps(chunk, ensure_ascii=False) + "\n")
        with self._graph_path.open("w", encoding="utf-8") as fh:
            json.dump(self._graph, fh, ensure_ascii=False)
        meta = {
            "n_chunks": len(self._chunks),
            "n_docs": len({c["doc_id"] for c in self._chunks}),
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
        }
        with self._index_path.open("w", encoding="utf-8") as fh:
            json.dump(meta, fh, ensure_ascii=False)

    def _write_empty(self) -> None:
        self._chunks = []
        self._graph = {}
        self._chunk_terms = []
        self._chunk_embeddings = None
        self._chunks_path.write_text("", encoding="utf-8")
        self._graph_path.write_text("{}", encoding="utf-8")
        self._index_path.write_text('{"n_chunks": 0}', encoding="utf-8")


# ----------------------------------------------------------------------
# Chunking (fixed-token approximation of LightRAG's chunker)
# ----------------------------------------------------------------------
def _fixed_chunk(text: str, chunk_size: int, overlap: int) -> list[str]:
    tokens = text.split()
    if not tokens:
        return []
    if chunk_size <= 0:
        return [text]
    if overlap >= chunk_size:
        overlap = max(0, chunk_size - 1)
    step = max(1, chunk_size - overlap)
    chunks: list[str] = []
    i = 0
    while i < len(tokens):
        piece = tokens[i : i + chunk_size]
        if piece:
            chunks.append(" ".join(piece))
        if i + chunk_size >= len(tokens):
            break
        i += step
    return chunks


# ----------------------------------------------------------------------
# Tokenization
# ----------------------------------------------------------------------
_WORD_RE = re.compile(r"[A-Za-z\u0370-\u03FF0-9]+", re.UNICODE)
_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for",
    "is", "are", "was", "were", "be", "been", "being", "with",
    "as", "by", "at", "from", "that", "this", "these", "those",
    "it", "its", "και", "να", "το", "τη", "την", "ο", "η", "οι",
    "τα", "των", "της", "του", "σε", "με", "για", "από", "που",
    "είναι", "στο", "στη",
}


def _tokenize(text: str) -> list[str]:
    tokens = _WORD_RE.findall(text.lower())
    return [t for t in tokens if len(t) > 2 and t not in _STOPWORDS]


# ----------------------------------------------------------------------
# Co-occurrence graph (fallback for LLM-based entity/relation graph)
# ----------------------------------------------------------------------
def _build_cooccurrence_graph(
    chunks: list[dict[str, Any]], window: int = 3
) -> dict[str, dict[str, float]]:
    graph: dict[str, dict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    for chunk in chunks:
        tokens = _tokenize(chunk["text"])
        for i, tok in enumerate(tokens):
            for j in range(i + 1, min(i + 1 + window, len(tokens))):
                other = tokens[j]
                if other == tok:
                    continue
                graph[tok][other] += 1.0
                graph[other][tok] += 1.0
    return {k: dict(v) for k, v in graph.items()}


# ----------------------------------------------------------------------
# Keyword extraction (fallback for LLM-based keyword extraction)
# ----------------------------------------------------------------------
def _extract_keywords(query: str, max_kw: int = 8) -> list[str]:
    tokens = _tokenize(query)
    if not tokens:
        return []
    return [tok for tok, _ in Counter(tokens).most_common(max_kw)]


# ----------------------------------------------------------------------
# Math helpers
# ----------------------------------------------------------------------
def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    if len(a) != len(b):
        n = min(len(a), len(b))
        a, b = a[:n], b[:n]
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _term_overlap_score(q: Counter, d: Counter) -> float:
    if not q or not d:
        return 0.0
    common = set(q) & set(d)
    if not common:
        return 0.0
    num = sum(q[t] * d[t] for t in common)
    den = (sum(q.values()) * sum(d.values())) ** 0.5
    return num / den if den else 0.0


def _fallback_embed(text: str, dim: int = 256) -> list[float]:
    vec = [0.0] * dim
    for tok in _tokenize(text):
        h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
        vec[h % dim] += 1.0
    return vec
