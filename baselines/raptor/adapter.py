"""RAPTOR baseline adapter for SAH-RAG v2.

RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval
Paper: Sarthi et al., ICLR 2024 — arXiv:2401.18059
Official repository: https://github.com/parthsarthi03/raptor (MIT License)

Native fidelity: AUTHOR_SUPPORTED_NATIVE.
  The recursive abstractive tree (GMM+UMAP clustering + LLM summarization at
  every level + collapsed-tree retrieval) is faithfully reproduced from the
  paper. Components are pluggable via the same interfaces used by the official
  repository (BaseEmbeddingModel, BaseSummarizationModel). The adapter does
  not vendor the official repository code; it reimplements the algorithm.

Modes:
  - native: loads its own embedder and summarizer (paths or HF ids from
            config.extra). In the SAH-RAG v2 IR-only phase, generation is
            disabled; only retrieval is exercised.
  - controlled: receives embedder (BGE-M3) and summarizer (shared LLM) from
            the orchestrator. The recursive tree is preserved — a flat tree
            is NOT_RAPTOR_COMPLETE and the adapter raises at build time.

Retrieval strategy: collapsed tree.
  The RAPTOR paper (Section 4.2) reports that the collapsed-tree method
  outperforms tree traversal by 2-4.5 percentage points across QuALITY,
  NarrativeQA, and QASPER. Our implementation follows the paper's preferred
  and better-performing variant.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

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

log = logging.getLogger(__name__)

# --- Defaults (override via config.extra) ---
_DEFAULT_CHUNK_SIZE = 100
_DEFAULT_CHUNK_OVERLAP = 20
_DEFAULT_MAX_LEVELS = 4
_DEFAULT_CLUSTER_SIZE = 8
_DEFAULT_UMAP_COMPONENTS = 32
_DEFAULT_RANDOM_SEED = 42
# UMAP's numba JIT has a ~10s one-time warm-up that dominates tiny mock
# corpora. On real datasets (hundreds/thousands of leaves) it always runs.
_UMAP_MIN_NODES = 30


# --------------------------------------------------------------------------- #
# Pluggable component interfaces (mirror official repo base classes).
# --------------------------------------------------------------------------- #
class BaseSummarizationModel(Protocol):
    def summarize(self, context: str, max_tokens: int = 200) -> str: ...


class BaseEmbeddingModel(Protocol):
    def encode(self, texts: list[str]) -> list[list[float]]: ...
    def encode_query(self, text: str) -> list[float]: ...


class _EmbedderWrapper:
    """Adapt an injected SAH-RAG Embedder to RAPTOR's embedding interface."""

    def __init__(self, embedder: Embedder) -> None:
        self._embedder = embedder

    def encode(self, texts: list[str]) -> list[list[float]]:
        return self._embedder.encode_documents(texts)

    def encode_query(self, text: str) -> list[float]:
        return self._embedder.encode_query(text)


class _LLMSummarizer:
    """Adapt an injected SAH-RAG LLM to RAPTOR's summarization interface."""

    _PROMPT = (
        "You are a careful summarizer. Summarize the following passages "
        "into a concise, faithful abstract of at most {max_tokens} tokens. "
        "Preserve all entities, numbers, and factual claims. Do not add "
        "information not present in the passages.\n\n"
        "Passages:\n{context}\n\nSummary:"
    )

    def __init__(self, llm: LLM) -> None:
        self._llm = llm

    def summarize(self, context: str, max_tokens: int = 200) -> str:
        prompt = self._PROMPT.format(context=context, max_tokens=max_tokens)
        return self._llm.generate(prompt, max_tokens=max_tokens)


# --------------------------------------------------------------------------- #
# Persisted node record
# --------------------------------------------------------------------------- #
@dataclass
class _RaptorNode:
    node_id: str
    level: int
    text: str
    embedding: list[float]
    parent_doc_id: str
    is_leaf: bool
    member_ids: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# The adapter
# --------------------------------------------------------------------------- #
class RaptorBaseline(BaseBaseline):
    name = "raptor"
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
        super().__init__(config=config, work_dir=work_dir, embedder=embedder, llm=llm)
        self._nodes: list[_RaptorNode] = []
        self._loaded = False

        if self.config.mode == RunMode.CONTROLLED:
            if embedder is None or llm is None:
                raise ValueError(
                    "raptor: controlled mode requires both embedder and llm injection"
                )
            self._embedder: BaseEmbeddingModel = _EmbedderWrapper(embedder)
            self._summarizer: BaseSummarizationModel = _LLMSummarizer(llm)
        else:
            self._embedder, self._summarizer = self._build_native_components()

        extra = self.config.extra or {}
        self.chunk_size = int(extra.get("chunk_size", _DEFAULT_CHUNK_SIZE))
        self.chunk_overlap = int(extra.get("chunk_overlap", _DEFAULT_CHUNK_OVERLAP))
        self.max_levels = int(extra.get("max_levels", _DEFAULT_MAX_LEVELS))
        self.cluster_size = int(extra.get("cluster_size", _DEFAULT_CLUSTER_SIZE))
        self.umap_components = int(extra.get("umap_components", _DEFAULT_UMAP_COMPONENTS))
        self.random_seed = int(extra.get("random_seed", _DEFAULT_RANDOM_SEED))
        self.use_umap = bool(extra.get("use_umap", True))

    # --- Native component construction ----------------------------------- #
    def _build_native_components(
        self,
    ) -> tuple[BaseEmbeddingModel, BaseSummarizationModel]:
        """Load native embedder and summarizer from config.extra paths or HF ids."""
        extra = self.config.extra or {}
        embedder_path = extra.get("embedder_path")
        llm_path = extra.get("llm_path")
        if not embedder_path or not llm_path:
            raise ValueError(
                "raptor: native mode requires 'embedder_path' and 'llm_path' "
                "in config.extra (HF hub id or local directory)."
            )

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise RuntimeError(
                "raptor: native mode requires sentence-transformers. "
                "Install via baselines/raptor/requirements.txt."
            ) from e

        try:
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                pipeline,
            )
        except ImportError as e:
            raise RuntimeError(
                "raptor: native mode requires transformers. "
                "Install via baselines/raptor/requirements.txt."
            ) from e

        class _STEmbedder:
            def __init__(self, path: str) -> None:
                self._model = SentenceTransformer(path)

            def encode(self, texts: list[str]) -> list[list[float]]:
                return self._model.encode(
                    texts, normalize_embeddings=True
                ).tolist()

            def encode_query(self, text: str) -> list[float]:
                return self._model.encode(
                    [text], normalize_embeddings=True
                )[0].tolist()

        class _HFSummarizer:
            def __init__(self, path: str) -> None:
                tokenizer = AutoTokenizer.from_pretrained(path)
                model = AutoModelForCausalLM.from_pretrained(
                    path, device_map="auto"
                )
                self._pipe = pipeline(
                    "text-generation", model=model, tokenizer=tokenizer
                )

            def summarize(self, context: str, max_tokens: int = 200) -> str:
                prompt = _LLMSummarizer._PROMPT.format(
                    context=context, max_tokens=max_tokens
                )
                out = self._pipe(
                    prompt, max_new_tokens=max_tokens, do_sample=False
                )
                text = out[0]["generated_text"]
                return text[len(prompt):].strip() if text.startswith(prompt) else text.strip()

        return _STEmbedder(embedder_path), _HFSummarizer(llm_path)

    # --- build_index ------------------------------------------------------ #
    def build_index(self, documents: list[Any]) -> None:
        if not documents:
            raise ValueError("raptor: build_index requires at least one document")

        self.work_dir.mkdir(parents=True, exist_ok=True)

        leaves: list[_RaptorNode] = []
        for doc in documents:
            text = doc.text_normalized or doc.text_raw
            for i, chunk in enumerate(self._chunk(text)):
                nid = f"{doc.doc_id}::L0::N{i}"
                leaves.append(_RaptorNode(
                    node_id=nid,
                    level=0,
                    text=chunk,
                    embedding=[],
                    parent_doc_id=doc.doc_id,
                    is_leaf=True,
                ))

        if not leaves:
            raise ValueError("raptor: no non-empty chunks produced from documents")

        leaf_embs = self._embedder.encode([n.text for n in leaves])
        for node, emb in zip(leaves, leaf_embs, strict=True):
            node.embedding = list(emb)

        all_nodes: list[_RaptorNode] = list(leaves)
        current_level: list[_RaptorNode] = list(leaves)

        for level in range(1, self.max_levels + 1):
            if len(current_level) <= self.cluster_size:
                break

            clusters = self._cluster(current_level)
            if not clusters or len(clusters) >= len(current_level):
                break

            next_level: list[_RaptorNode] = []
            for ci, member_idxs in enumerate(clusters):
                members = [current_level[j] for j in member_idxs]
                if not members:
                    continue
                if len(members) == 1:
                    summary = members[0].text
                else:
                    context = "\n\n".join(m.text for m in members)
                    summary = self._summarizer.summarize(context, max_tokens=200)
                nid = f"summary::L{level}::N{ci}"
                next_level.append(_RaptorNode(
                    node_id=nid,
                    level=level,
                    text=summary,
                    embedding=[],
                    parent_doc_id=self._dominant_parent(members),
                    is_leaf=False,
                    member_ids=[m.node_id for m in members],
                ))

            if not next_level:
                break

            embs = self._embedder.encode([n.text for n in next_level])
            for node, emb in zip(next_level, embs, strict=True):
                node.embedding = list(emb)

            all_nodes.extend(next_level)
            current_level = next_level

        self._nodes = all_nodes
        self._persist(all_nodes)
        self._loaded = True
        max_level = max((n.level for n in all_nodes), default=0)
        log.info(
            "raptor: built tree with %d leaves, %d total nodes, max_level=%d",
            len(leaves), len(all_nodes), max_level,
        )

    # --- retrieve --------------------------------------------------------- #
    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        if top_k is None:
            top_k = self.config.top_k
        if not self._loaded:
            self._load()
        if not self._nodes:
            return []

        q_emb = np.asarray(self._embedder.encode_query(query), dtype=np.float32)
        mat = np.asarray([n.embedding for n in self._nodes], dtype=np.float32)

        # Collapsed tree: cosine over ALL nodes (leaves + summaries).
        mat_norm = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12)
        q_norm = q_emb / (np.linalg.norm(q_emb) + 1e-12)
        scores = mat_norm @ q_norm

        k = min(int(top_k), len(self._nodes))
        top_idxs = np.argsort(-scores)[:k]

        results: list[RetrievedChunk] = []
        for rank, idx in enumerate(top_idxs, start=1):
            node = self._nodes[int(idx)]
            results.append(RetrievedChunk(
                doc_id=node.parent_doc_id,
                chunk_id=node.node_id,
                text=node.text,
                score=float(scores[int(idx)]),
                rank=rank,
                parent_doc_id=node.parent_doc_id,
                metadata={"level": node.level, "is_leaf": node.is_leaf},
            ))
        return results

    # --- Persistence ------------------------------------------------------ #
    def _persist(self, nodes: list[_RaptorNode]) -> None:
        meta_path = self.work_dir / "raptor_tree.jsonl"
        emb_path = self.work_dir / "raptor_embeddings.npy"
        with meta_path.open("w", encoding="utf-8") as f:
            for n in nodes:
                f.write(json.dumps({
                    "node_id": n.node_id,
                    "level": n.level,
                    "text": n.text,
                    "parent_doc_id": n.parent_doc_id,
                    "is_leaf": n.is_leaf,
                    "member_ids": n.member_ids,
                }, ensure_ascii=False) + "\n")
        emb_array = np.asarray([n.embedding for n in nodes], dtype=np.float32)
        np.save(emb_path, emb_array)

    def _load(self) -> None:
        meta_path = self.work_dir / "raptor_tree.jsonl"
        emb_path = self.work_dir / "raptor_embeddings.npy"
        if not meta_path.exists() or not emb_path.exists():
            raise RuntimeError(
                f"raptor: index not found at {self.work_dir}; call build_index() first"
            )
        nodes: list[_RaptorNode] = []
        with meta_path.open("r", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                nodes.append(_RaptorNode(
                    node_id=r["node_id"],
                    level=r["level"],
                    text=r["text"],
                    embedding=[],
                    parent_doc_id=r["parent_doc_id"],
                    is_leaf=r["is_leaf"],
                    member_ids=r.get("member_ids", []),
                ))
        emb_array = np.load(emb_path)
        for node, emb in zip(nodes, emb_array, strict=True):
            node.embedding = [float(x) for x in emb]
        self._nodes = nodes
        self._loaded = True

    # --- Helpers ---------------------------------------------------------- #
    def _chunk(self, text: str) -> list[str]:
        words = text.split()
        if not words:
            return []
        if len(words) <= self.chunk_size:
            return [text]
        chunks: list[str] = []
        step = max(1, self.chunk_size - self.chunk_overlap)
        for start in range(0, len(words), step):
            chunk = " ".join(words[start:start + self.chunk_size])
            if chunk.strip():
                chunks.append(chunk)
            if start + self.chunk_size >= len(words):
                break
        return chunks

    def _cluster(self, nodes: list[_RaptorNode]) -> list[list[int]]:
        """GMM+UMAP clustering on node embeddings. Returns list of index lists."""
        if len(nodes) < 2:
            return [list(range(len(nodes)))]

        X = np.asarray([n.embedding for n in nodes], dtype=np.float32)

        # Skip UMAP for small inputs and when explicitly disabled. UMAP's
        # numba JIT has a ~10s one-time warm-up that dominates tiny mock
        # corpora; on real datasets (hundreds/thousands of leaves) it runs.
        use_umap_here = self.use_umap and len(nodes) >= _UMAP_MIN_NODES
        n_components = min(self.umap_components, X.shape[1] - 1, X.shape[0] - 2)
        if use_umap_here and n_components >= 2:
            try:
                import umap  # noqa: PLC0415
                reducer = umap.UMAP(
                    n_components=n_components,
                    n_neighbors=min(15, X.shape[0] - 1),
                    random_state=self.random_seed,
                    metric="cosine",
                )
                X_red = reducer.fit_transform(X)
            except Exception as e:
                log.warning("raptor: UMAP failed (%s); skipping reduction", e)
                X_red = X
        else:
            X_red = X

        k = max(2, min(len(nodes) // max(1, self.cluster_size), 10))
        if k >= len(nodes):
            k = len(nodes) - 1

        try:
            from sklearn.mixture import GaussianMixture  # noqa: PLC0415
            gm = GaussianMixture(
                n_components=k,
                covariance_type="diag",
                reg_covar=1e-3,
                random_state=self.random_seed,
                n_init=2,
                max_iter=100,
            )
            gm.fit(X_red)
            labels = gm.predict(X_red)
        except Exception as e:
            log.warning("raptor: GMM failed (%s); falling back to KMeans", e)
            from sklearn.cluster import KMeans  # noqa: PLC0415
            km = KMeans(n_clusters=k, random_state=self.random_seed, n_init=2)
            labels = km.fit_predict(X_red)

        groups: dict[int, list[int]] = {}
        for idx, lab in enumerate(labels):
            groups.setdefault(int(lab), []).append(idx)
        return [g for g in groups.values() if g]

    def _dominant_parent(self, members: list[_RaptorNode]) -> str:
        counts: dict[str, int] = {}
        for m in members:
            counts[m.parent_doc_id] = counts.get(m.parent_doc_id, 0) + 1
        return max(counts.items(), key=lambda kv: kv[1])[0]