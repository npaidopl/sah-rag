"""ColBERTv2 baseline adapter for SAH-RAG v2 (native-only).

Implements the frozen `BaseBaseline` contract (`baselines/base.py`).

Reference
---------
Paper : ColBERTv2: Effective and Efficient Retrieval via Lightweight
        Late Interaction (arXiv:2112.01488, NAACL 2022)
Code  : https://github.com/stanford-futuredata/ColBERT
Ckpt  : https://huggingface.co/colbert-ir/colbertv2.0
License: MIT (code + checkpoint)

Design
------
* Native mode only. Controlled mode is architecturally incompatible with
  ColBERTv2 because BGE-M3 is a dense embedder, while ColBERTv2 requires
  a token-level BERT-based encoder for its late-interaction scoring.
* The adapter chunks each `CanonicalDocument.text_normalized` into fixed
  character windows (with overlap), writes a `collection.tsv` and a
  `mapping.json` under `self.work_dir`, then builds a native PLAID index
  via the official `colbert-ai` package.
* Retrieval decodes passage IDs back to `RetrievedChunk` records with
  contiguous ranks 1..k.

Persistence layout (all under `self.work_dir`)
----------------------------------------------
work_dir/
├── collection.tsv                 # ColBERTv2 collection: "<int>\t<text>"
├── mapping.json                   # {int_id: {chunk_id, doc_id, text}}
└── colbert_experiments/
    └── colbertv2_native/
        └── indexes/
            └── corpus/            # native PLAID index
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

from baselines.base import (
    BaseBaseline,
    BaselineConfig,
    BaselineStatus,
    RetrievedChunk,
    RunMode,
)

if TYPE_CHECKING:
    from sah_rag.canonical.models import CanonicalDocument

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Defaults (can be overridden via config.extra)
# ---------------------------------------------------------------------------

DEFAULT_CHECKPOINT = "colbert-ir/colbertv2.0"
DEFAULT_DOC_MAXLEN = 512
DEFAULT_NBITS = 2
DEFAULT_NCELLS = 1
DEFAULT_CHUNK_SIZE = 1800       # chars (~450 tokens at ~4 chars/token)
DEFAULT_CHUNK_OVERLAP = 200     # chars (~50 tokens)
DEFAULT_EXPERIMENT = "colbertv2_native"
DEFAULT_INDEX_NAME = "corpus"


class ColBERTv2Baseline(BaseBaseline):
    """ColBERTv2 late-interaction retrieval baseline (native mode only)."""

    # ---- Identity (contract: class-level attributes) ----
    name = "colbertv2"
    version = "0.1.0"
    status = BaselineStatus.EXECUTABLE
    supports_native = True
    supports_controlled = False  # architectural incompatibility with BGE-M3
    is_llm_based = False         # retrieval-only, no generation

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def __init__(
        self,
        *,
        config: BaselineConfig,
        work_dir: Path,
        embedder: Any = None,
        llm: Any = None,
    ) -> None:
        super().__init__(
            config=config,
            work_dir=work_dir,
            embedder=embedder,
            llm=llm,
        )

        extra = config.extra or {}

        # ---- Frozen configuration ----
        self._checkpoint: str = str(
            extra.get("checkpoint", DEFAULT_CHECKPOINT)
        )
        self._doc_maxlen: int = int(
            extra.get("doc_maxlen", DEFAULT_DOC_MAXLEN)
        )
        self._nbits: int = int(extra.get("nbits", DEFAULT_NBITS))
        self._ncells: int = int(extra.get("ncells", DEFAULT_NCELLS))
        self._chunk_size: int = int(
            extra.get("chunk_size", DEFAULT_CHUNK_SIZE)
        )
        self._chunk_overlap: int = int(
            extra.get("chunk_overlap", DEFAULT_CHUNK_OVERLAP)
        )
        self._experiment: str = str(
            extra.get("experiment", DEFAULT_EXPERIMENT)
        )
        self._index_name: str = str(
            extra.get("index_name", DEFAULT_INDEX_NAME)
        )

        if self._chunk_overlap >= self._chunk_size:
            raise ValueError(
                f"chunk_overlap ({self._chunk_overlap}) must be < "
                f"chunk_size ({self._chunk_size})"
            )

        # ---- Persistence paths (all under work_dir) ----
        self._collection_path: Path = self.work_dir / "collection.tsv"
        self._mapping_path: Path = self.work_dir / "mapping.json"
        self._experiments_root: Path = self.work_dir / "colbert_experiments"

        # ---- Runtime state ----
        self._built: bool = False
        self._mapping: dict[str, dict[str, Any]] | None = None
        self._searcher: Any = None
        self._searcher_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
        """Split `text` into overlapping character windows.

        Simple, deterministic, and reliable. Each chunk (except possibly
        the last) has length `chunk_size`; consecutive chunks overlap by
        `overlap` characters.
        """
        if not text or not text.strip():
            return []
        if len(text) <= chunk_size:
            return [text]
        if overlap >= chunk_size:
            raise ValueError(
                f"overlap ({overlap}) must be < chunk_size ({chunk_size})"
            )
        step = chunk_size - overlap
        chunks: list[str] = []
        for i in range(0, len(text), step):
            chunk = text[i : i + chunk_size]
            if chunk:
                chunks.append(chunk)
            if i + chunk_size >= len(text):
                break
        return chunks

    @staticmethod
    def _sanitize_for_tsv(text: str) -> str:
        """Replace tabs and newlines so the row stays on one TSV line."""
        return text.replace("\t", " ").replace("\r", " ").replace("\n", " ")

    # ------------------------------------------------------------------
    # build_index
    # ------------------------------------------------------------------

    def build_index(self, documents: list["CanonicalDocument"]) -> None:
        """Chunk documents, write collection.tsv + mapping.json, build index."""
        if not documents:
            raise ValueError("build_index received an empty document list.")

        # 1) Build collection.tsv and mapping in one pass.
        mapping: dict[str, dict[str, Any]] = {}
        next_int_id = 0
        n_docs_indexed = 0

        with self._collection_path.open("w", encoding="utf-8") as coll_fh:
            for doc in documents:
                text = doc.text_normalized or ""
                chunks = self._chunk_text(
                    text, self._chunk_size, self._chunk_overlap
                )
                if not chunks:
                    logger.warning(
                        "Skipping empty document: doc_id=%s", doc.doc_id
                    )
                    continue

                for ci, chunk in enumerate(chunks):
                    chunk_id = f"{doc.doc_id}_c{ci}"
                    row = self._sanitize_for_tsv(chunk)
                    coll_fh.write(f"{next_int_id}\t{row}\n")
                    mapping[str(next_int_id)] = {
                        "chunk_id": chunk_id,
                        "doc_id": doc.doc_id,
                        "text": chunk,
                    }
                    next_int_id += 1
                n_docs_indexed += 1

        if not mapping:
            raise ValueError(
                "No indexable documents: every input document had empty text."
            )

        # 2) Persist mapping.
        with self._mapping_path.open("w", encoding="utf-8") as fh:
            json.dump(mapping, fh, ensure_ascii=False, indent=2)

        # 3) Build the native ColBERTv2 index.
        from colbert.infra import ColBERTConfig, Run, RunConfig
        from colbert import Indexer

        self._experiments_root.mkdir(parents=True, exist_ok=True)

        with Run().context(
            RunConfig(
                nranks=1,
                root=str(self._experiments_root),
                experiment=self._experiment,
            )
        ):
            colbert_cfg = ColBERTConfig(
                doc_maxlen=self._doc_maxlen,
                nbits=self._nbits,
                ncells=self._ncells,
            )
            indexer = Indexer(
                checkpoint=self._checkpoint,
                config=colbert_cfg,
            )
            indexer.index(
                name=self._index_name,
                collection=str(self._collection_path),
                overwrite=True,
            )

        self._mapping = mapping
        self._built = True
        logger.info(
            "ColBERTv2 index built: %d chunks from %d/%d documents "
            "(checkpoint=%s, doc_maxlen=%d, nbits=%d, ncells=%d)",
            len(mapping),
            n_docs_indexed,
            len(documents),
            self._checkpoint,
            self._doc_maxlen,
            self._nbits,
            self._ncells,
        )

    # ------------------------------------------------------------------
    # retrieve
    # ------------------------------------------------------------------

    def retrieve(
        self, query: str, top_k: int | None = None
    ) -> list[RetrievedChunk]:
        """Return top-k chunks for `query`, with contiguous ranks 1..k."""
        if not self._built:
            raise RuntimeError(
                "build_index() must be called before retrieve()."
            )
        if not query or not query.strip():
            return []

        k = top_k if top_k is not None else self.config.top_k
        if k <= 0:
            return []

        mapping = self._get_mapping()
        searcher = self._ensure_searcher()

        # colbert-ai returns a namedtuple (pids, ranks, scores).
        results = searcher.search(query, k=k)
        pids = list(results.pids) if hasattr(results, "pids") else list(results[0])
        scores = (
            list(results.scores)
            if hasattr(results, "scores")
            else list(results[2])
        )

        chunks: list[RetrievedChunk] = []
        for i, pid in enumerate(pids):
            entry = mapping.get(str(int(pid)))
            if entry is None:
                logger.warning(
                    "ColBERTv2 returned unknown passage id: pid=%s", pid
                )
                continue
            chunks.append(
                RetrievedChunk(
                    doc_id=entry["doc_id"],
                    chunk_id=entry["chunk_id"],
                    text=entry["text"],
                    score=float(scores[i]),
                    rank=len(chunks) + 1,  # contiguous 1..k by construction
                    parent_doc_id=None,
                    metadata={"colbert_pid": int(pid)},
                )
            )

        return chunks

    # ------------------------------------------------------------------
    # Lazy internal helpers
    # ------------------------------------------------------------------

    def _get_mapping(self) -> dict[str, dict[str, Any]]:
        if self._mapping is None:
            if not self._mapping_path.exists():
                raise RuntimeError(
                    f"mapping.json not found at {self._mapping_path}; "
                    "was build_index() run?"
                )
            with self._mapping_path.open("r", encoding="utf-8") as fh:
                self._mapping = json.load(fh)
        return self._mapping

    def _ensure_searcher(self) -> Any:
        if self._searcher is not None:
            return self._searcher
        with self._searcher_lock:
            if self._searcher is not None:
                return self._searcher
            from colbert.infra import Run, RunConfig
            from colbert import Searcher

            with Run().context(
                RunConfig(
                    nranks=1,
                    root=str(self._experiments_root),
                    experiment=self._experiment,
                )
            ):
                self._searcher = Searcher(
                    index=self._index_name,
                    collection=str(self._collection_path),
                )
            return self._searcher

    # ------------------------------------------------------------------
    # Optional hook
    # ------------------------------------------------------------------

    def cleanup(self) -> None:
        """Release the Searcher (GPU memory) after evaluation."""
        self._searcher = None