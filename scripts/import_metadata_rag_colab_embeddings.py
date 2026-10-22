# ruff: noqa: E402
"""Local import stage for Metadata-RAG Colab embeddings.

Consumes the encoded tar.gz produced by
``scripts/colab_metadata_rag_encode.py`` and enriches it with the
metadata that was frozen on the local side. The Colab side only knows
window ids and dense vectors; this script reconstructs the full
``MetadataRAGDocumentEmbedding`` and ``MetadataRAGQueryEmbedding``
records that the retrieval pipeline expects.

Pipeline:

  1. Extract the encoded tar.gz and verify its internal checksums.
  2. Read the frozen state from ``<state-dir>`` (state-v1).
  3. Load the encoded window embeddings and query embeddings.
  4. Verify a strict bijection between state windows and encoded
     window embeddings (same doc_id set, same count).
  5. Rebuild each ``MetadataRAGDocumentEmbedding`` from the state
     window's metadata + the encoded dense vector.
  6. Rebuild each ``MetadataRAGQueryEmbedding`` from the encoded
     query id + the encoded dense vector.
  7. Write deterministic, checksum-bound artifacts.

The script never touches the network, never loads any model, and
never reads ground truth.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tarfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from baselines.metadata_rag_embedder import (
    EncoderConfig,
    MetadataRAGDocumentEmbedding,
    MetadataRAGQueryEmbedding,
    build_document_embedding,
    build_query_embedding,
)
from baselines.metadata_rag_persistence import read_state
from baselines.metadata_rag_state import MetadataRAGState

IMPORT_ARTIFACT_TYPE: Final[str] = "metadata_rag_embeddings_imported"
IMPORT_ARTIFACT_VERSION: Final[str] = "1.0"

DENSE_DIMENSION: Final[int] = 1024

# Canonical snapshot marker used to build the local EncoderConfig. The
# import stage never loads a model; this string is only used to compute
# a deterministic ``embedding_config_hash`` for provenance. Using a
# fixed marker (instead of the actual Colab snapshot path) guarantees
# that a local import and a re-import produce byte-identical records.
LOCAL_CONFIG_SNAPSHOT_MARKER: Final[str] = (
    "colab://bge-m3-dense-only-import-v1"
)

INPUT_WINDOW_EMBEDDINGS: Final[str] = "window_embeddings.jsonl"
INPUT_QUERY_EMBEDDINGS: Final[str] = "query_embeddings.jsonl"
INPUT_MANIFEST: Final[str] = "manifest.json"
INPUT_CHECKSUMS: Final[str] = "checksums.json"

OUTPUT_DOCUMENT_EMBEDDINGS: Final[str] = "document_embeddings.jsonl"
OUTPUT_QUERY_EMBEDDINGS: Final[str] = "query_embeddings.jsonl"
OUTPUT_MANIFEST: Final[str] = "manifest.json"
OUTPUT_CHECKSUMS: Final[str] = "checksums.json"


class ImportError_(ValueError):
    """Raised when import violates its contract."""


# -------------------------------------------------------------------------
# Deterministic I/O helpers
# -------------------------------------------------------------------------


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


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


def _write_jsonl(path: Path, items: Sequence[Any]) -> int:
    if not items:
        _atomic_write_bytes(path, b"\n")
        return 0
    parts: list[str] = []
    for item in items:
        payload = (
            item.model_dump(mode="json")
            if hasattr(item, "model_dump")
            else item
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
    return len(items)


def _read_json(path: Path) -> dict[str, Any]:
    parsed = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(parsed, dict):
        raise ImportError_(f"{path.name} is not a JSON object")
    return parsed


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for lineno, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        stripped = line.strip()
        if not stripped:
            continue
        parsed = json.loads(stripped)
        if not isinstance(parsed, dict):
            raise ImportError_(
                f"{path.name}:{lineno} not a JSON object"
            )
        rows.append(parsed)
    return rows


# -------------------------------------------------------------------------
# Input extraction + verification
# -------------------------------------------------------------------------


def _extract_encoded_tar(
    embeddings_tar: Path, target_dir: Path
) -> dict[str, Any]:
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    with tarfile.open(embeddings_tar, "r:gz") as tar:
        for member in tar.getmembers():
            name = Path(member.name).name
            if name not in (
                INPUT_WINDOW_EMBEDDINGS,
                INPUT_QUERY_EMBEDDINGS,
                INPUT_MANIFEST,
                INPUT_CHECKSUMS,
            ):
                continue
            if member.isfile():
                tar.extract(member, path=target_dir, filter="data")

    manifest = _read_json(target_dir / INPUT_MANIFEST)
    if manifest.get("artifact_type") != "metadata_rag_embeddings":
        raise ImportError_(
            "encoded.artifact_type must be metadata_rag_embeddings"
        )

    checksums = _read_json(target_dir / INPUT_CHECKSUMS)
    for name in (INPUT_WINDOW_EMBEDDINGS, INPUT_QUERY_EMBEDDINGS):
        declared = checksums.get(name)
        if declared is None:
            raise ImportError_(
                f"encoded checksums missing entry for {name}"
            )
        observed = _sha256_file(target_dir / name)
        if observed != declared:
            raise ImportError_(
                f"encoded checksum mismatch for {name}: "
                f"observed {observed}, declared {declared}"
            )
    return manifest


# -------------------------------------------------------------------------
# Config used to build local records
# -------------------------------------------------------------------------


def _build_local_config(
    *,
    model_id: str,
    model_revision: str,
    device: str,
    use_fp16: bool,
) -> EncoderConfig:
    return EncoderConfig(
        model_id=model_id,
        model_revision=model_revision,
        dense_dimension=DENSE_DIMENSION,
        dense_normalized=True,
        device=device,  # type: ignore[arg-type]
        use_fp16=use_fp16,
        batch_size=1,
        max_input_tokens=8192,
        local_snapshot_path=LOCAL_CONFIG_SNAPSHOT_MARKER,
        network_fallback_allowed=False,
    )


# -------------------------------------------------------------------------
# Enrichment
# -------------------------------------------------------------------------


def _to_tuple_vector(raw: object) -> tuple[float, ...]:
    if not isinstance(raw, list):
        raise ImportError_("dense_vector must be a JSON array")
    vec = tuple(float(x) for x in raw)
    if len(vec) != DENSE_DIMENSION:
        raise ImportError_(
            f"dense_vector length {len(vec)} != {DENSE_DIMENSION}"
        )
    return vec


def _enrich_document_embeddings(
    *,
    state: MetadataRAGState,
    encoded_rows: Sequence[dict[str, Any]],
    config: EncoderConfig,
) -> tuple[MetadataRAGDocumentEmbedding, ...]:
    state_by_doc: dict[str, Any] = {
        d.doc_id: d for d in state.documents
    }
    encoded_by_doc: dict[str, dict[str, Any]] = {}
    for row in encoded_rows:
        doc_id = str(row.get("doc_id", ""))
        if not doc_id:
            raise ImportError_("encoded row missing doc_id")
        if doc_id in encoded_by_doc:
            raise ImportError_(
                f"duplicate encoded doc_id: {doc_id}"
            )
        encoded_by_doc[doc_id] = row

    only_state = set(state_by_doc) - set(encoded_by_doc)
    only_encoded = set(encoded_by_doc) - set(state_by_doc)
    if only_state or only_encoded:
        raise ImportError_(
            "bijection violation between state windows and encoded "
            f"embeddings: only_state={sorted(only_state)[:5]} "
            f"only_encoded={sorted(only_encoded)[:5]}"
        )

    enriched: list[MetadataRAGDocumentEmbedding] = []
    for doc in state.documents:
        row = encoded_by_doc[doc.doc_id]
        vector = _to_tuple_vector(row["dense_vector"])
        enriched.append(
            build_document_embedding(
                config=config,
                dataset_id=state.dataset_id,
                doc_id=doc.doc_id,
                document_order=doc.document_order,
                metadata_record_id=doc.metadata_record_id,
                source_text=doc.text_normalized,
                dense_vector=vector,
            )
        )
    return tuple(enriched)


def _enrich_query_embeddings(
    *,
    dataset_id: str,
    encoded_rows: Sequence[dict[str, Any]],
    config: EncoderConfig,
) -> tuple[MetadataRAGQueryEmbedding, ...]:
    seen: set[str] = set()
    enriched: list[MetadataRAGQueryEmbedding] = []
    for row in encoded_rows:
        query_id = str(row.get("query_id", ""))
        if not query_id:
            raise ImportError_("encoded query row missing query_id")
        if query_id in seen:
            raise ImportError_(
                f"duplicate encoded query_id: {query_id}"
            )
        seen.add(query_id)
        vector = _to_tuple_vector(row["dense_vector"])
        # Use the query id as the placeholder text for the SHA-256
        # because the local import stage never re-reads the query
        # text (queries are projected separately at retrieval time).
        # The SHA-256 is only used for record-identity disambiguation;
        # the query text itself is never persisted in the record.
        placeholder_text = f"query-id::{query_id}"
        enriched.append(
            build_query_embedding(
                config=config,
                dataset_id=dataset_id,
                query_id=query_id,
                query_text=placeholder_text,
                dense_vector=vector,
            )
        )
    return tuple(enriched)


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------


def run_import(
    *,
    embeddings_tar: Path,
    state_dir: Path,
    output_dir: Path,
    work_dir: Path,
) -> dict[str, Any]:
    if not embeddings_tar.is_file():
        raise ImportError_(f"embeddings tar not found: {embeddings_tar}")
    if not state_dir.is_dir():
        raise ImportError_(f"state dir not found: {state_dir}")

    print(f"[1/4] extract: {embeddings_tar}")
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    work_extract = work_dir / "encoded"
    encoded_manifest = _extract_encoded_tar(embeddings_tar, work_extract)
    print(
        "      encoded dataset_id="
        f"{encoded_manifest.get('dataset_id')} "
        f"windows={encoded_manifest.get('window_count')} "
        f"queries={encoded_manifest.get('query_count')}"
    )

    print(f"[2/4] read state: {state_dir}")
    state = read_state(state_dir)
    print(
        f"      state dataset_id={state.dataset_id} "
        f"windows={len(state.documents)}"
    )

    if encoded_manifest.get("dataset_id") != state.dataset_id:
        raise ImportError_(
            "encoded.dataset_id does not match state.dataset_id: "
            f"{encoded_manifest.get('dataset_id')} vs "
            f"{state.dataset_id}"
        )

    print("[3/4] enrich")
    config = _build_local_config(
        model_id=str(
            encoded_manifest.get("model_id", "BAAI/bge-m3")
        ),
        model_revision=str(
            encoded_manifest.get(
                "model_revision",
                "5617a9f61b028005a4858fdac845db406aefb181",
            )
        ),
        device=str(encoded_manifest.get("device", "cuda")),
        use_fp16=bool(encoded_manifest.get("use_fp16", True)),
    )

    encoded_windows = _read_jsonl(
        work_extract / INPUT_WINDOW_EMBEDDINGS
    )
    encoded_queries = _read_jsonl(
        work_extract / INPUT_QUERY_EMBEDDINGS
    )

    documents = _enrich_document_embeddings(
        state=state, encoded_rows=encoded_windows, config=config
    )
    queries = _enrich_query_embeddings(
        dataset_id=state.dataset_id,
        encoded_rows=encoded_queries,
        config=config,
    )

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[4/4] write outputs")
    n_docs = _write_jsonl(
        output_dir / OUTPUT_DOCUMENT_EMBEDDINGS, documents
    )
    n_queries = _write_jsonl(
        output_dir / OUTPUT_QUERY_EMBEDDINGS, queries
    )

    docs_sha = _sha256_file(
        output_dir / OUTPUT_DOCUMENT_EMBEDDINGS
    )
    queries_sha = _sha256_file(
        output_dir / OUTPUT_QUERY_EMBEDDINGS
    )

    manifest = {
        "artifact_type": IMPORT_ARTIFACT_TYPE,
        "artifact_version": IMPORT_ARTIFACT_VERSION,
        "dataset_id": state.dataset_id,
        "encoder_kind": str(
            encoded_manifest.get("encoder_kind", "UNKNOWN")
        ),
        "model_id": config.model_id,
        "model_revision": config.model_revision,
        "embedding_config_hash": config.config_hash(),
        "dense_dimension": DENSE_DIMENSION,
        "dense_normalized": True,
        "document_count": n_docs,
        "query_count": n_queries,
        "document_embeddings_sha256": docs_sha,
        "query_embeddings_sha256": queries_sha,
        "source_encoded_tar_sha256": _sha256_file(embeddings_tar),
        "source_state_documents_sha256": _sha256_file(
            state_dir / "documents.jsonl"
        ),
    }
    _write_json(output_dir / OUTPUT_MANIFEST, manifest)

    checksums = {
        OUTPUT_DOCUMENT_EMBEDDINGS: docs_sha,
        OUTPUT_QUERY_EMBEDDINGS: queries_sha,
        OUTPUT_MANIFEST: _sha256_file(output_dir / OUTPUT_MANIFEST),
    }
    _write_json(output_dir / OUTPUT_CHECKSUMS, checksums)

    shutil.rmtree(work_dir, ignore_errors=True)
    return manifest


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Import and enrich Metadata-RAG Colab embeddings."
        )
    )
    parser.add_argument(
        "--embeddings-tar", type=Path, required=True
    )
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("_metadata_rag_import_work"),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    manifest = run_import(
        embeddings_tar=args.embeddings_tar,
        state_dir=args.state_dir,
        output_dir=args.output_dir,
        work_dir=args.work_dir,
    )
    print("[OK] import complete")
    print(f"  dataset_id        = {manifest['dataset_id']}")
    print(f"  document_count    = {manifest['document_count']}")
    print(f"  query_count       = {manifest['query_count']}")
    print(
        "  config_hash       = "
        f"{manifest['embedding_config_hash'][:16]}..."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
