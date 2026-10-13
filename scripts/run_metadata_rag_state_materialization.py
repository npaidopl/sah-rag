# ruff: noqa: E402
"""Metadata-RAG L2 - State materialization + Colab bundle preparation.

For each of the four real datasets (clerk, cuad, greek_bar_bench, lara):

  1. Stream canonical documents.
  2. Build MetadataRAGState with the pinned BGE-M3 windowing splitter
     (amendment 04). The tokenizer is loaded locally, offline, from
     the frozen BGE-M3 snapshot.
  3. Persist the state under <out>/<dataset>/state-v1/.
  4. Build a self-contained Colab bundle with window texts and query
     texts under <out>/<dataset>/colab_bundle/.
  5. Package the bundle into <out>/<dataset>/colab_bundle.tar.gz.

The script never touches the BGE-M3 model, never encodes, and never
reads ground truth. It only materializes deterministic window spans
and the flat text bundles the Colab-side encoder will consume.

Resume-safe: if state-v1/manifest.json exists with status SUCCEEDED
and the bundle is intact, the dataset is skipped.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tarfile
import time
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Final

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rag_system.baselines.metadata_rag_adapter import (
    ADAPTER_VERSION,
    MetadataRAGProjectedQuery,
    build_indexed_documents,
    project_queries,
)
from rag_system.baselines.metadata_rag_persistence import (
    write_state,
)
from rag_system.baselines.metadata_rag_state import (
    MetadataRAGState,
    build_state,
)
from rag_system.baselines.metadata_rag_windowing import (
    WINDOW_CONTENT_TOKENS,
    build_bge_m3_window_splitter,
)
from rag_system.canonical.loaders import (
    iter_canonical_documents,
    iter_canonical_examples,
)

DEFAULT_OUTPUT_ROOT: Final[Path] = (
    REPO_ROOT / "outputs" / "external_metadata_rag"
)
DEFAULT_SNAPSHOT: Final[Path] = Path(
    "D:/rag_work/hf_cache/models--BAAI--bge-m3/snapshots/"
    "5617a9f61b028005a4858fdac845db406aefb181"
)
EXPECTED_DATASETS: tuple[str, ...] = (
    "clerk",
    "cuad",
    "greek_bar_bench",
    "lara",
)

BUNDLE_ARTIFACT_TYPE: Final[str] = "metadata_rag_colab_bundle"
BUNDLE_ARTIFACT_VERSION: Final[str] = "1.0"


class MetadataRAGMaterializationError(ValueError):
    """Raised when materialization violates its contract."""


# -------------------------------------------------------------------------
# I/O helpers
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


def _write_jsonl_atomic(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    parts: list[str] = []
    count = 0
    for row in rows:
        parts.append(
            json.dumps(
                row,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            )
        )
        count += 1
    payload = ("\n".join(parts) + "\n").encode("utf-8")
    _atomic_write_bytes(path, payload)
    return count


def _read_json(path: Path) -> dict[str, Any]:
    parsed = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(parsed, dict):
        raise MetadataRAGMaterializationError(
            f"{path.name} is not a JSON object"
        )
    return parsed


def _is_already_done(dataset_dir: Path) -> bool:
    manifest_path = dataset_dir / "state-v1" / "manifest.json"
    bundle_path = dataset_dir / "colab_bundle.tar.gz"
    bundle_manifest = dataset_dir / "colab_bundle" / "manifest.json"
    if not (
        manifest_path.is_file()
        and bundle_path.is_file()
        and bundle_manifest.is_file()
    ):
        return False
    try:
        state_manifest = _read_json(manifest_path)
        bundle_meta = _read_json(bundle_manifest)
    except (OSError, json.JSONDecodeError):
        return False
    return (
        state_manifest.get("artifact_type") == "metadata_rag_state"
        and bundle_meta.get("artifact_type") == BUNDLE_ARTIFACT_TYPE
    )


# -------------------------------------------------------------------------
# State + bundle
# -------------------------------------------------------------------------


def _build_state_for_dataset(
    *,
    dataset: str,
    snapshot_path: Path,
) -> MetadataRAGState:
    documents = tuple(iter_canonical_documents(dataset))
    if not documents:
        raise MetadataRAGMaterializationError(
            f"canonical documents are empty for {dataset}"
        )
    splitter = build_bge_m3_window_splitter(
        snapshot_path=snapshot_path,
        max_content_tokens=WINDOW_CONTENT_TOKENS,
    )
    indexed = build_indexed_documents(
        documents,
        dataset_id=dataset,
        splitter=splitter,
    )
    return build_state(indexed, dataset_id=dataset)


def _build_bundle(
    *,
    state: MetadataRAGState,
    projected_queries: Sequence[MetadataRAGProjectedQuery],
    bundle_dir: Path,
) -> dict[str, Any]:
    bundle_dir.mkdir(parents=True, exist_ok=True)

    window_rows: list[dict[str, Any]] = [
        {"doc_id": d.doc_id, "text": d.text_normalized}
        for d in state.documents
    ]
    windows_path = bundle_dir / "windows.jsonl"
    n_windows = _write_jsonl_atomic(windows_path, window_rows)

    query_rows: list[dict[str, Any]] = [
        {"query_id": q.query_id, "text": q.query_text}
        for q in projected_queries
    ]
    queries_path = bundle_dir / "queries.jsonl"
    n_queries = _write_jsonl_atomic(queries_path, query_rows)

    windows_sha = _sha256_file(windows_path)
    queries_sha = _sha256_file(queries_path)

    manifest: dict[str, Any] = {
        "artifact_type": BUNDLE_ARTIFACT_TYPE,
        "artifact_version": BUNDLE_ARTIFACT_VERSION,
        "dataset_id": state.dataset_id,
        "window_count": n_windows,
        "query_count": n_queries,
        "windows_sha256": windows_sha,
        "queries_sha256": queries_sha,
        "state_version": state.state_version,
        "adapter_version": ADAPTER_VERSION,
    }
    _write_json(bundle_dir / "manifest.json", manifest)

    checksums = {
        "windows.jsonl": windows_sha,
        "queries.jsonl": queries_sha,
        "manifest.json": _sha256_file(bundle_dir / "manifest.json"),
    }
    _write_json(bundle_dir / "checksums.json", checksums)
    return manifest


def _package_bundle(bundle_dir: Path, target_tar: Path) -> None:
    tmp = target_tar.with_name(target_tar.name + ".tmp")
    with tarfile.open(tmp, "w:gz") as tar:
        for name in (
            "manifest.json",
            "windows.jsonl",
            "queries.jsonl",
            "checksums.json",
        ):
            tar.add(bundle_dir / name, arcname=name)
    tmp.replace(target_tar)


# -------------------------------------------------------------------------
# Per-dataset driver
# -------------------------------------------------------------------------


def _materialize_one(
    *,
    dataset: str,
    output_root: Path,
    snapshot_path: Path,
    force: bool,
) -> dict[str, Any]:
    dataset_dir = output_root / dataset
    if not force and _is_already_done(dataset_dir):
        existing = _read_json(
            dataset_dir / "state-v1" / "manifest.json"
        )
        return {
            "dataset": dataset,
            "status": "SKIPPED_ALREADY_DONE",
            "document_count": existing.get("document_count"),
        }

    if force and dataset_dir.exists():
        shutil.rmtree(dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.monotonic()
    state = _build_state_for_dataset(
        dataset=dataset, snapshot_path=snapshot_path
    )
    state_secs = time.monotonic() - t0

    write_state(state, dataset_dir / "state-v1")

    examples = tuple(iter_canonical_examples(dataset))
    projected = project_queries(examples)

    bundle_dir = dataset_dir / "colab_bundle"
    bundle_manifest = _build_bundle(
        state=state,
        projected_queries=projected,
        bundle_dir=bundle_dir,
    )

    bundle_tar = dataset_dir / "colab_bundle.tar.gz"
    _package_bundle(bundle_dir, bundle_tar)

    summary = {
        "dataset": dataset,
        "status": "SUCCEEDED",
        "document_count": len(state.documents),
        "query_count": len(projected),
        "state_seconds": round(state_secs, 3),
        "bundle_bytes": bundle_tar.stat().st_size,
        "bundle_sha256": _sha256_file(bundle_tar),
        "bundle_manifest": bundle_manifest,
    }
    _write_json(dataset_dir / "materialization_summary.json", summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize Metadata-RAG state and Colab bundles for the "
            "four real datasets."
        )
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=DEFAULT_SNAPSHOT,
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=list(EXPECTED_DATASETS),
    )
    parser.add_argument(
        "--force",
        action="store_true",
    )
    args = parser.parse_args(argv)

    if not args.snapshot.is_dir():
        raise SystemExit(
            f"BGE-M3 snapshot missing: {args.snapshot}"
        )

    summaries: list[dict[str, Any]] = []
    for dataset in args.datasets:
        if dataset not in EXPECTED_DATASETS:
            raise SystemExit(f"unknown dataset: {dataset}")
        summary = _materialize_one(
            dataset=dataset,
            output_root=args.output_root,
            snapshot_path=args.snapshot,
            force=args.force,
        )
        summaries.append(summary)
        print(
            f"[{summary['status']}] {dataset}: "
            f"docs={summary.get('document_count')} "
            f"queries={summary.get('query_count')}"
        )

    print("[OK] Metadata-RAG state materialization")
    for s in summaries:
        print(
            f"  {s['dataset']:<18} status={s['status']:<22} "
            f"windows={s.get('document_count')} "
            f"queries={s.get('query_count')}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
