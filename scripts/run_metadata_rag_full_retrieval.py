# ruff: noqa: E402
"""Metadata-RAG L5 - Full retrieval across the 4-dataset scope.

For each dataset:

  1. Load the persisted state (state-v1/).
  2. Load enriched document embeddings (embeddings-v1/).
  3. Build the exact in-memory dense reference index.
  4. Load enriched query embeddings (embeddings-v1/).
  5. Load query texts from the frozen bundle (colab_bundle/).
  6. Verify a strict bijection between state windows, document
     embeddings, query embeddings, and query texts.
  7. Run batch retrieval: query_text -> constraints -> candidate
     filter -> exact dense cosine -> metadata-semantic fusion ->
     ranked items.
  8. Write retrieval-v1/results.jsonl + manifest + checksums.

Ground truth is NEVER loaded. Only query_id and query_text are read.
Resume-safe: if retrieval-v1/manifest.json exists SUCCEEDED, skip.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rag_system.baselines.metadata_rag_adapter import (
    MetadataRAGProjectedQuery,
)
from rag_system.baselines.metadata_rag_embedder import (
    MetadataRAGDocumentEmbedding,
    MetadataRAGQueryEmbedding,
)
from rag_system.baselines.metadata_rag_persistence import read_state
from rag_system.baselines.metadata_rag_reference_index import (
    ReferenceDenseIndex,
    build_reference_dense_index,
)
from rag_system.baselines.metadata_rag_retrieval import (
    QueryRetrievalResult,
    RetrievalConfig,
    retrieve_batch,
)
from rag_system.baselines.metadata_rag_state import MetadataRAGState

DEFAULT_OUTPUT_ROOT: Final[Path] = (
    REPO_ROOT / "outputs" / "external_metadata_rag"
)
EXPECTED_DATASETS: Final[tuple[str, ...]] = (
    "clerk",
    "cuad",
    "greek_bar_bench",
    "lara",
)

RETRIEVAL_ARTIFACT_TYPE: Final[str] = "metadata_rag_retrieval"
RETRIEVAL_ARTIFACT_VERSION: Final[str] = "1.0"

STATE_DIR: Final[str] = "state-v1"
EMBEDDINGS_DIR: Final[str] = "embeddings-v1"
BUNDLE_DIR: Final[str] = "colab_bundle"
RETRIEVAL_DIR: Final[str] = "retrieval-v1"

EMBEDDINGS_DOCUMENTS_FILE: Final[str] = "document_embeddings.jsonl"
EMBEDDINGS_QUERIES_FILE: Final[str] = "query_embeddings.jsonl"
BUNDLE_QUERIES_FILE: Final[str] = "queries.jsonl"

OUTPUT_RESULTS_FILE: Final[str] = "results.jsonl"
OUTPUT_MANIFEST_FILE: Final[str] = "manifest.json"
OUTPUT_CHECKSUMS_FILE: Final[str] = "checksums.json"


class MetadataRAGRetrievalRunError(ValueError):
    """Raised when the full-retrieval stage violates its contract."""


# -------------------------------------------------------------------------
# I/O
# -------------------------------------------------------------------------


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _atomic_write_bytes(target: Path, payload: bytes) -> None:
    tmp = target.with_name(target.name + ".tmp")
    tmp.write_bytes(payload)
    tmp.replace(target)


def _write_json(path: Path, obj: Any) -> None:
    text = json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    _atomic_write_bytes(path, (text + "\n").encode("utf-8"))


def _write_jsonl(path: Path, rows: Sequence[Any]) -> int:
    if not rows:
        _atomic_write_bytes(path, b"\n")
        return 0
    parts: list[str] = []
    for row in rows:
        payload = (
            row.model_dump(mode="json")
            if hasattr(row, "model_dump")
            else row
        )
        parts.append(
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            )
        )
    _atomic_write_bytes(path, ("\n".join(parts) + "\n").encode("utf-8"))
    return len(rows)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise MetadataRAGRetrievalRunError(f"missing file: {path}")
    rows: list[dict[str, Any]] = []
    for lineno, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        stripped = line.strip()
        if not stripped:
            continue
        parsed = json.loads(stripped)
        if not isinstance(parsed, dict):
            raise MetadataRAGRetrievalRunError(
                f"{path.name}:{lineno} not a JSON object"
            )
        rows.append(parsed)
    return rows


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise MetadataRAGRetrievalRunError(f"missing file: {path}")
    parsed = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(parsed, dict):
        raise MetadataRAGRetrievalRunError(
            f"{path.name} is not a JSON object"
        )
    return parsed


def _is_already_done(dataset_dir: Path) -> bool:
    manifest_path = (
        dataset_dir / RETRIEVAL_DIR / OUTPUT_MANIFEST_FILE
    )
    if not manifest_path.is_file():
        return False
    try:
        manifest = _read_json(manifest_path)
    except (OSError, json.JSONDecodeError):
        return False
    return manifest.get("artifact_type") == RETRIEVAL_ARTIFACT_TYPE


# -------------------------------------------------------------------------
# Loading
# -------------------------------------------------------------------------


def _load_state(dataset_dir: Path) -> MetadataRAGState:
    return read_state(dataset_dir / STATE_DIR)


def _load_document_embeddings(
    dataset_dir: Path,
) -> tuple[MetadataRAGDocumentEmbedding, ...]:
    rows = _read_jsonl(
        dataset_dir / EMBEDDINGS_DIR / EMBEDDINGS_DOCUMENTS_FILE
    )
    return tuple(
        MetadataRAGDocumentEmbedding.model_validate(r) for r in rows
    )


def _load_query_embeddings(
    dataset_dir: Path,
) -> tuple[MetadataRAGQueryEmbedding, ...]:
    rows = _read_jsonl(
        dataset_dir / EMBEDDINGS_DIR / EMBEDDINGS_QUERIES_FILE
    )
    return tuple(
        MetadataRAGQueryEmbedding.model_validate(r) for r in rows
    )


def _load_query_texts(
    dataset_dir: Path,
) -> dict[str, str]:
    rows = _read_jsonl(dataset_dir / BUNDLE_DIR / BUNDLE_QUERIES_FILE)
    out: dict[str, str] = {}
    for row in rows:
        query_id = str(row.get("query_id", ""))
        text = str(row.get("text", ""))
        if not query_id or not text:
            raise MetadataRAGRetrievalRunError(
                "bundle query row missing query_id or text"
            )
        if query_id in out:
            raise MetadataRAGRetrievalRunError(
                f"duplicate query_id in bundle: {query_id}"
            )
        out[query_id] = text
    return out


# -------------------------------------------------------------------------
# Bijection checks
# -------------------------------------------------------------------------


def _verify_bijections(
    *,
    state: MetadataRAGState,
    document_embeddings: Sequence[MetadataRAGDocumentEmbedding],
    query_embeddings: Sequence[MetadataRAGQueryEmbedding],
    query_texts: dict[str, str],
) -> None:
    state_doc_ids = {d.doc_id for d in state.documents}
    emb_doc_ids = {e.doc_id for e in document_embeddings}
    if state_doc_ids != emb_doc_ids:
        only_state = state_doc_ids - emb_doc_ids
        only_emb = emb_doc_ids - state_doc_ids
        raise MetadataRAGRetrievalRunError(
            "state <-> document embeddings bijection violated: "
            f"only_state={sorted(only_state)[:5]} "
            f"only_emb={sorted(only_emb)[:5]}"
        )

    emb_query_ids = {e.query_id for e in query_embeddings}
    text_query_ids = set(query_texts)
    if emb_query_ids != text_query_ids:
        only_emb = emb_query_ids - text_query_ids
        only_text = text_query_ids - emb_query_ids
        raise MetadataRAGRetrievalRunError(
            "query embeddings <-> query texts bijection violated: "
            f"only_emb={sorted(only_emb)[:5]} "
            f"only_text={sorted(only_text)[:5]}"
        )


def _project_queries(
    *,
    query_embeddings: Sequence[MetadataRAGQueryEmbedding],
    query_texts: dict[str, str],
) -> tuple[
    tuple[MetadataRAGProjectedQuery, ...],
    tuple[tuple[float, ...], ...],
]:
    queries: list[MetadataRAGProjectedQuery] = []
    vectors: list[tuple[float, ...]] = []
    for emb in query_embeddings:
        text = query_texts[emb.query_id]
        queries.append(
            MetadataRAGProjectedQuery(
                query_id=emb.query_id, query_text=text
            )
        )
        vectors.append(emb.dense_vector)
    return tuple(queries), tuple(vectors)


# -------------------------------------------------------------------------
# One dataset
# -------------------------------------------------------------------------


def _run_one(
    *,
    dataset: str,
    output_root: Path,
    force: bool,
) -> dict[str, Any]:
    dataset_dir = output_root / dataset
    if not force and _is_already_done(dataset_dir):
        existing = _read_json(
            dataset_dir / RETRIEVAL_DIR / OUTPUT_MANIFEST_FILE
        )
        return {
            "dataset": dataset,
            "status": "SKIPPED_ALREADY_DONE",
            "query_count": existing.get("query_count"),
        }

    t0 = time.monotonic()

    print(f"[1/5] load state: {dataset}")
    state = _load_state(dataset_dir)
    print(f"      windows: {len(state.documents)}")

    print("[2/5] load embeddings")
    document_embeddings = _load_document_embeddings(dataset_dir)
    query_embeddings = _load_query_embeddings(dataset_dir)
    print(
        f"      document_embeddings={len(document_embeddings)} "
        f"query_embeddings={len(query_embeddings)}"
    )

    print("[3/5] load query texts from bundle")
    query_texts = _load_query_texts(dataset_dir)
    print(f"      query_texts={len(query_texts)}")

    _verify_bijections(
        state=state,
        document_embeddings=document_embeddings,
        query_embeddings=query_embeddings,
        query_texts=query_texts,
    )

    print("[4/5] build reference index + retrieve")
    index: ReferenceDenseIndex = build_reference_dense_index(
        state=state, embeddings=document_embeddings
    )
    queries, vectors = _project_queries(
        query_embeddings=query_embeddings, query_texts=query_texts
    )
    config = RetrievalConfig()
    results: tuple[QueryRetrievalResult, ...] = retrieve_batch(
        queries=queries,
        index=index,
        query_vectors=vectors,
        config=config,
    )

    # Sanity: every query must return at least one ranked item.
    empty = [r.query_id for r in results if not r.ranked_items]
    if empty:
        raise MetadataRAGRetrievalRunError(
            f"{len(empty)} queries returned zero ranked items; "
            f"first: {empty[:5]}"
        )

    if force and (dataset_dir / RETRIEVAL_DIR).exists():
        shutil.rmtree(dataset_dir / RETRIEVAL_DIR)
    retrieval_dir = dataset_dir / RETRIEVAL_DIR
    retrieval_dir.mkdir(parents=True, exist_ok=True)

    print("[5/5] write outputs")
    n = _write_jsonl(retrieval_dir / OUTPUT_RESULTS_FILE, results)
    results_sha = _sha256_file(retrieval_dir / OUTPUT_RESULTS_FILE)

    elapsed = time.monotonic() - t0
    manifest = {
        "artifact_type": RETRIEVAL_ARTIFACT_TYPE,
        "artifact_version": RETRIEVAL_ARTIFACT_VERSION,
        "dataset_id": state.dataset_id,
        "window_count": len(state.documents),
        "query_count": n,
        "retrieval_config": {
            "alpha_metadata": config.alpha_metadata,
            "alpha_semantic": config.alpha_semantic,
            "min_match_ratio": config.min_match_ratio,
            "document_limit": config.document_limit,
        },
        "source_state_documents_sha256": _sha256_file(
            dataset_dir / STATE_DIR / "documents.jsonl"
        ),
        "source_document_embeddings_sha256": _sha256_file(
            dataset_dir
            / EMBEDDINGS_DIR
            / EMBEDDINGS_DOCUMENTS_FILE
        ),
        "source_query_embeddings_sha256": _sha256_file(
            dataset_dir
            / EMBEDDINGS_DIR
            / EMBEDDINGS_QUERIES_FILE
        ),
        "source_bundle_queries_sha256": _sha256_file(
            dataset_dir / BUNDLE_DIR / BUNDLE_QUERIES_FILE
        ),
        "results_sha256": results_sha,
        "elapsed_seconds": round(elapsed, 3),
        "generation_enabled": False,
        "ground_truth_loaded": False,
    }
    _write_json(retrieval_dir / OUTPUT_MANIFEST_FILE, manifest)

    checksums = {
        OUTPUT_RESULTS_FILE: results_sha,
        OUTPUT_MANIFEST_FILE: _sha256_file(
            retrieval_dir / OUTPUT_MANIFEST_FILE
        ),
    }
    _write_json(retrieval_dir / OUTPUT_CHECKSUMS_FILE, checksums)

    return {
        "dataset": dataset,
        "status": "SUCCEEDED",
        "window_count": len(state.documents),
        "query_count": n,
        "elapsed_seconds": round(elapsed, 3),
        "results_sha256": results_sha,
    }


# -------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Metadata-RAG L5 full retrieval over the four datasets."
        )
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=list(EXPECTED_DATASETS),
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    summaries: list[dict[str, Any]] = []
    for dataset in args.datasets:
        if dataset not in EXPECTED_DATASETS:
            raise SystemExit(f"unknown dataset: {dataset}")
        summary = _run_one(
            dataset=dataset,
            output_root=args.output_root,
            force=args.force,
        )
        summaries.append(summary)
        print(
            f"[{summary['status']}] {dataset}: "
            f"windows={summary.get('window_count')} "
            f"queries={summary.get('query_count')} "
            f"elapsed={summary.get('elapsed_seconds')}s"
        )
    print("[OK] Metadata-RAG full retrieval")
    for s in summaries:
        print(
            f"  {s['dataset']:<18} status={s['status']:<22} "
            f"windows={s.get('window_count')} "
            f"queries={s.get('query_count')}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
