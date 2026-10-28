"""DISRetrieval baseline adapter for SAH-RAG v2 — precompute-then-retrieve.

Reference:
    Chen et al., "Beyond Chunking: Discourse-Aware Hierarchical Retrieval
    for Long Document Question Answering", arXiv:2506.06313 (ACL 2026).

Native status: BLOCKED_MISSING_BACKBONE.
    The DISRetrieval bundled checkpoint head weights exist, but the
    fine-tuned XLNet backbone (xlm_rst_tuned) is not published anywhere
    in the official repository. Paper-exact reproduction is therefore
    impossible without training and licensing the original data.

Controlled status: CONTROLLED_ADAPTATION (active).
    We use the upstream E-NNRSTParser (Apache-2.0) as the neural RST
    parser, combined with the DISRetrieval architecture-defining logic:
      1. RST-parse each document into a discourse tree.
      2. LLM-summarize internal nodes bottom-up.
      3. Embed node summaries and leaf text.
      4. rest_topk: top-down traversal, keep top-k children per level.
      5. Leaf expansion: collect surviving leaves per document.
      6. Evidence assembly: rank leaves by query similarity.

    This is NOT the paper-native DISRetrieval. The parser head weights
    are different, the backbone is different, the training data is
    different. The comparison is architecture-level, not reproduction.
    Declared as CONTROLLED_ADAPTATION in the SAH-RAG registry.

Two execution paths:
    (A) Precomputed trees (production):
        build_index() loads per-document trees from
        D:/rag_work/disretrieval_trees/<dataset>/doc_<id>.json.
        Fast. No parser or summarizer loads at query time.

    (B) On-the-fly build (fallback / smoke test):
        build_index() parses documents directly. Requires the
        E-NNRSTParser code, its checkpoint, and an injected LLM for
        summarization. Only allowed when extra.allow_on_the_fly_build=true.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
from pathlib import Path
from typing import Any, Protocol

from baselines.base import (
    BaselineConfig,
    BaselineStatus,
    BaseBaseline,
    RetrievedChunk,
    RunMode,
)
from sah_rag.canonical.models import CanonicalDocument

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Protocols for optional parser / summarizer (injected or loaded lazily)
# ---------------------------------------------------------------------------

class ParserProtocol(Protocol):
    """Minimal parser contract used by the adapter."""

    def parse(self, text: str) -> dict[str, Any]:
        """Return a discourse tree as a nested dict with keys:
        id, text, summary, children (list), level (int)."""


class SummarizerProtocol(Protocol):
    """Minimal summarizer contract used by the adapter."""

    def summarize(self, context: str, max_tokens: int = 200) -> str:
        ...


# ---------------------------------------------------------------------------
# Stub components (TEST ONLY — enabled via extra.use_stub_components=true)
# ---------------------------------------------------------------------------

class _StubEmbedder:
    """Deterministic hash-based embedder. TEST ONLY."""

    name = "stub_embedder"

    @staticmethod
    def _hash(text: str, dim: int = 16) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [digest[i] / 255.0 for i in range(dim)]

    def encode_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._hash(t) for t in texts]

    def encode_query(self, text: str) -> list[float]:
        return self._hash(text)


class _StubLLM:
    """Fixed-return summarizer. TEST ONLY."""

    name = "stub_llm"

    def generate(self, prompt: str, **kwargs: Any) -> str:
        return "summary"


class _SimpleParagraphParser:
    """Paragraph splitter + binary grouping.

    TEST ONLY. A heuristic replacement of the neural RST parser violates
    the DISRetrieval architecture identity and must never be used in
    reported experiments.
    """

    def parse(self, text: str) -> dict[str, Any]:
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        if len(paragraphs) <= 1:
            paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
        if not paragraphs:
            paragraphs = [text.strip() or "empty"]

        leaves: list[dict[str, Any]] = [
            {"id": f"leaf_{i}", "text": p, "summary": p, "children": [], "level": 0}
            for i, p in enumerate(paragraphs)
        ]

        current = leaves
        level = 1
        while len(current) > 1:
            nxt: list[dict[str, Any]] = []
            for i in range(0, len(current), 2):
                group = current[i : i + 2]
                joined = " ".join(c["text"] for c in group)
                nxt.append(
                    {
                        "id": f"node_L{level}_{i}",
                        "text": joined,
                        "summary": joined,
                        "children": group,
                        "level": level,
                    }
                )
            current = nxt
            level += 1

        return current[0]


# ---------------------------------------------------------------------------
# Lazy-loaded real components (used only in on-the-fly path)
# ---------------------------------------------------------------------------

class _RealRSTParser:
    """Wrapper around DISRetrieval/E-NNRSTParser.

    Loads the parser lazily so that the adapter can be imported in the
    project venv (torch 2.14 / transformers 5.x) without triggering the
    incompatible code path. The actual parse() call only works when the
    dedicated venv (torch 2.2.2 / transformers 4.42.3) is active.
    """

    def __init__(self, disretrieval_root: Path, config_path: Path) -> None:
        self._root = disretrieval_root
        self._config_path = config_path
        self._parser = None

    def _ensure_loaded(self) -> None:
        if self._parser is not None:
            return
        import sys
        import torch as _torch

        # CPU-friendly torch.load patch (checkpoint was saved on CUDA)
        _orig_load = _torch.load

        def _cpu_load(*args, **kwargs):
            kwargs.setdefault("map_location", "cpu")
            kwargs.setdefault("weights_only", False)
            return _orig_load(*args, **kwargs)

        _torch.load = _cpu_load  # type: ignore[assignment]
        try:
            sys.path.insert(0, str(self._root))
            sys.path.insert(0, str(self._root / "rst_parser"))
            from rst_parser.driver.RSTParser import RSTParser  # type: ignore

            self._parser = RSTParser(config_file=str(self._config_path), use_cuda=False)
        finally:
            _torch.load = _orig_load  # type: ignore[assignment]

    def parse(self, text: str) -> dict[str, Any]:
        # Placeholder: full parse is handled by scripts/precompute_trees.py
        # which invokes RSTParser.start_parsing on a JSON file of EDUs.
        raise NotImplementedError(
            "Real RST parsing is available only through "
            "scripts/precompute_trees.py in the dedicated venv. "
            "The adapter in production consumes precomputed trees."
        )


class _LLMSummarizerWrapper:
    """Adapter around the injected LLM for internal-node summarization."""

    _PROMPT = (
        "Summarize the following passage in one concise sentence. "
        "Preserve entities, numbers, and factual claims.\n\n"
        "{context}\n\nSummary:"
    )

    def __init__(self, llm: Any) -> None:
        self._llm = llm

    def summarize(self, context: str, max_tokens: int = 200) -> str:
        try:
            return self._llm.generate(
                self._PROMPT.format(context=context[:2000]),
                max_tokens=max_tokens,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("LLM summarization failed, using truncation: %s", exc)
            return context[:200]


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

class DisRetrievalBaseline(BaseBaseline):
    """DISRetrieval baseline adapter (controlled = E-NNRSTParser + this architecture)."""

    name = "disretrieval"
    version = "0.2.0"
    status = BaselineStatus.EXECUTABLE
    supports_native = False   # native blocked (missing backbone)
    supports_controlled = True
    is_llm_based = True       # LLM is architecture-defining at indexing

    def __init__(
        self,
        *,
        config: BaselineConfig,
        work_dir: Path,
        embedder: Any | None = None,
        llm: Any | None = None,
    ) -> None:
        super().__init__(config=config, work_dir=work_dir, embedder=embedder, llm=llm)

        # Per-document tree storage
        self._trees: dict[str, dict[str, Any]] = {}
        # Node embeddings cache (only used when node is embedded lazily)
        self._embedding_cache: dict[str, list[float]] = {}
        # Component backends (resolved lazily)
        self._parser: ParserProtocol | None = None
        self._summarizer: SummarizerProtocol | None = None
        self._embedder_backend: Any | None = None

    # ---- component resolution ----

    def _resolve_embedder(self) -> Any:
        if self._embedder_backend is not None:
            return self._embedder_backend
        if self.config.extra.get("use_stub_components", False):
            self._embedder_backend = _StubEmbedder()
        elif self.config.mode == RunMode.CONTROLLED:
            if self.embedder is None:
                raise RuntimeError("controlled mode requires injected embedder")
            self._embedder_backend = self.embedder
        else:
            raise RuntimeError(
                "Native mode is BLOCKED_MISSING_BACKBONE. Use controlled mode."
            )
        return self._embedder_backend

    def _resolve_summarizer(self) -> SummarizerProtocol:
        if self._summarizer is not None:
            return self._summarizer
        if self.config.extra.get("use_stub_components", False):
            self._summarizer = _LLMSummarizerWrapper(_StubLLM())
        else:
            if self.llm is None:
                raise RuntimeError(
                    "Controlled mode requires injected LLM for on-the-fly "
                    "tree building. Set precomputed_trees_root in extra to "
                    "avoid LLM usage at build time."
                )
            self._summarizer = _LLMSummarizerWrapper(self.llm)
        return self._summarizer

    def _resolve_parser(self) -> ParserProtocol:
        if self._parser is not None:
            return self._parser
        if self.config.extra.get("use_stub_components", False):
            self._parser = _SimpleParagraphParser()
            return self._parser

        raise RuntimeError(
            "On-the-fly parsing requires the dedicated venv "
            "(D:/rag_work/venvs/disretrieval_rst) and precomputed trees are "
            "the recommended production path. Set "
            "extra.precomputed_trees_root to use precomputed trees."
        )

    # ---- tree helpers ----

    @staticmethod
    def _collect_all_nodes(tree: dict[str, Any]) -> list[dict[str, Any]]:
        nodes = [tree]
        for child in tree.get("children", []):
            nodes.extend(DisRetrievalBaseline._collect_all_nodes(child))
        return nodes

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

    def _summarize_tree(
        self,
        node: dict[str, Any],
        summarizer: SummarizerProtocol,
    ) -> dict[str, Any]:
        children = node.get("children") or []
        if not children:
            return node
        new_children = [self._summarize_tree(c, summarizer) for c in children]
        joined = " ".join(c.get("text", "") for c in new_children)
        summary = summarizer.summarize(joined, max_tokens=100)
        return {**node, "children": new_children, "summary": summary}

    def _embed_tree(
        self,
        tree: dict[str, Any],
        embedder: Any,
    ) -> None:
        nodes = self._collect_all_nodes(tree)
        texts = [n.get("summary") or n.get("text", "") for n in nodes]
        if not texts:
            return
        vectors = embedder.encode_documents(texts)
        for node, vec in zip(nodes, vectors):
            self._embedding_cache[node["id"]] = [float(x) for x in vec]

    # ---- precomputed tree loading ----

    def _load_precomputed_trees(
        self,
        documents: list[CanonicalDocument],
    ) -> bool:
        root = self.config.extra.get("precomputed_trees_root")
        if not root:
            return False

        dataset_id_map = self.config.extra.get("dataset_id_map", {}) or {}
        dataset_id = documents[0].dataset if documents else None
        if not dataset_id or not documents:
            return False

        canonical = dataset_id_map.get(dataset_id, dataset_id)
        dataset_root = Path(root) / canonical
        if not dataset_root.exists():
            log.warning(
                "Precomputed trees not found at %s for dataset %s",
                dataset_root,
                dataset_id,
            )
            return False

        embedder = self._resolve_embedder()
        embedder_name = getattr(embedder, "name", "unknown")

        loaded = 0
        for doc in documents:
            path = dataset_root / f"doc_{doc.doc_id}.json"
            if not path.exists():
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:  # noqa: BLE001
                log.warning("Failed to load %s: %s", path, exc)
                continue

            tree = payload["tree"]
            embeddings = payload.get("embeddings", {})
            embedder_meta = payload.get("embedder", {})

            # If stored embeddings were produced with a different embedder,
            # re-embed lazily on demand (correctness over speed).
            if embedder_meta.get("name") == embedder_name and embeddings:
                self._trees[doc.doc_id] = tree
                for node_id, vec in embeddings.items():
                    self._embedding_cache[node_id] = [float(x) for x in vec]
            else:
                self._trees[doc.doc_id] = tree
                self._embed_tree(tree, embedder)

            loaded += 1

        if loaded:
            log.info("Loaded %d precomputed trees from %s", loaded, dataset_root)
            return True
        return False

    # ---- lifecycle ----

    def build_index(self, documents: list[CanonicalDocument]) -> None:
        self._trees = {}
        self._embedding_cache = {}

        # Path A: precomputed trees (fast, production)
        if self.config.extra.get("build_from_precomputed", True):
            if self._load_precomputed_trees(documents):
                self._write_manifest()
                return

        # Path B: on-the-fly (smoke test / stub fallback)
        if not (
            self.config.extra.get("allow_on_the_fly_build", False)
            or self.config.extra.get("use_stub_components", False)
        ):
            raise RuntimeError(
                "No precomputed trees available for this dataset and "
                "allow_on_the_fly_build=false. Generate trees with "
                "scripts/precompute_trees.py first."
            )

        parser = self._resolve_parser()
        summarizer = self._resolve_summarizer()
        embedder = self._resolve_embedder()

        for doc in documents:
            tree = parser.parse(doc.text_normalized)
            tree = self._summarize_tree(tree, summarizer)
            tree["doc_id"] = doc.doc_id
            self._trees[doc.doc_id] = tree
            self._embed_tree(tree, embedder)

        self._write_manifest()

    def _write_manifest(self) -> None:
        manifest = self.work_dir / "trees_manifest.json"
        payload = {
            "document_count": len(self._trees),
            "documents": list(self._trees.keys()),
            "config_name": self.config.name,
            "mode": self.config.mode.value,
            "native_fidelity": self.config.extra.get(
                "native_fidelity", "UNKNOWN"
            ),
        }
        manifest.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    # ---- rest_topk traversal ----

    def _rest_topk_leaves(
        self,
        node: dict[str, Any],
        query_vec: list[float],
        top_k: int,
        depth: int,
        max_depth: int,
    ) -> list[dict[str, Any]]:
        if depth >= max_depth or not node.get("children"):
            return [node]

        scored: list[tuple[dict[str, Any], float]] = []
        for child in node["children"]:
            emb = self._embedding_cache.get(child["id"], [])
            scored.append((child, self._cosine(query_vec, emb)))
        scored.sort(key=lambda x: x[1], reverse=True)

        out: list[dict[str, Any]] = []
        for child, _ in scored[:top_k]:
            out.extend(
                self._rest_topk_leaves(child, query_vec, top_k, depth + 1, max_depth)
            )
        return out

    # ---- retrieval ----

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        if top_k is None:
            top_k = self.config.top_k

        if not self._trees:
            return []

        embedder = self._resolve_embedder()
        retrieval_cfg = self.config.extra.get("retrieval", {}) or {}
        leaf_topk = int(retrieval_cfg.get("leaf_topk", 5))
        max_depth = int(retrieval_cfg.get("max_node_depth", 10000))

        query_vec = [float(x) for x in embedder.encode_query(query)]

        candidates: list[dict[str, Any]] = []
        for doc_id, tree in self._trees.items():
            leaves = self._rest_topk_leaves(tree, query_vec, top_k, 0, max_depth)
            for leaf in leaves:
                candidates.append({**leaf, "doc_id": doc_id})

        if not candidates:
            return []

        missing = [c for c in candidates if c["id"] not in self._embedding_cache]
        if missing:
            texts = [c.get("summary") or c.get("text", "") for c in missing]
            vectors = embedder.encode_documents(texts)
            for cand, vec in zip(missing, vectors):
                self._embedding_cache[cand["id"]] = [float(x) for x in vec]

        scored: list[tuple[dict[str, Any], float]] = []
        for cand in candidates:
            emb = self._embedding_cache.get(cand["id"], [])
            scored.append((cand, self._cosine(query_vec, emb)))
        scored.sort(key=lambda x: x[1], reverse=True)

        per_doc: dict[str, list[tuple[dict[str, Any], float]]] = {}
        for leaf, score in scored:
            per_doc.setdefault(leaf["doc_id"], []).append((leaf, score))

        capped: list[tuple[dict[str, Any], float]] = []
        for doc_items in per_doc.values():
            capped.extend(doc_items[:leaf_topk])
        capped.sort(key=lambda x: x[1], reverse=True)

        # Drop leaves with empty text (e.g. from span mismatch between
        # the RST parser output and the source sentence list). Chunks
        # without text are useless to the retrieval consumer and would
        # pollute metrics. Fallback: use the parent node text.
        filtered: list[tuple[dict[str, Any], float]] = []
        for leaf, score in capped:
            text = (leaf.get("text") or "").strip()
            if not text:
                text = (leaf.get("summary") or "").strip()
            if text:
                filtered.append(({**leaf, "text": text}, score))

        filtered = filtered[:top_k]

        results: list[RetrievedChunk] = []
        for rank, (leaf, score) in enumerate(filtered, start=1):
            results.append(
                RetrievedChunk(
                    doc_id=leaf["doc_id"],
                    chunk_id=leaf["id"],
                    text=leaf["text"],
                    score=float(score),
                    rank=rank,
                    parent_doc_id=leaf["doc_id"],
                    metadata={
                        "level": leaf.get("level", 0),
                        "native_fidelity": self.config.extra.get(
                            "native_fidelity", "UNKNOWN"
                        ),
                    },
                )
            )
        return results