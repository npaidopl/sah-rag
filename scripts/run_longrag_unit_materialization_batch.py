# ruff: noqa: E402
"""LongRAG L2 - Full text-unit materialization.

For each of the seven canonical datasets:
  canonical documents -> pinned BGE-M3 tokenization -> source segments
  -> long-unit construction (single_document_if_within_budget @ 4096)
  -> oversize split via BGE-M3 tokenizer windows
  -> units.jsonl + summary.json + checksums.json

Resume-safe per dataset: if units-v1/manifest.json is SUCCEEDED, skip.
Offline: pinned snapshot only, no network, no embeddings.
"""
from __future__ import annotations
# --- SAH-RAG path resolution (sanitized from local absolute paths) ---
import os as _os
from pathlib import Path as _Path
_SAH_RAG_REPO_ROOT = _Path(__file__).resolve().parents[1]
_SAH_RAG_PROJECT_ROOT = _os.environ.get(
    "SAH_RAG_PROJECT_ROOT",
    str(_SAH_RAG_REPO_ROOT.parent / "Project 1"),
)
_SAH_RAG_CANONICAL_ROOT = _os.environ.get(
    "SAH_RAG_CANONICAL_ROOT",
    str(_Path(_SAH_RAG_PROJECT_ROOT) / "processed" / "canonical"),
)
# ---------------------------------------------------------------------

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import yaml  # noqa: E402

from rag_system.baselines import (  # noqa: E402
    LongRAGSourceSegment,
    LongUnit,
    build_long_unit_id,
)
from rag_system.canonical.loaders import iter_canonical_documents  # noqa: E402

CANONICAL_ROOT = Path(
    _SAH_RAG_CANONICAL_ROOT
)
OUTPUT_ROOT = REPO_ROOT / "outputs" / "external_longrag"
BASELINE_AUDIT = REPO_ROOT / "baseline_audit"

EXPECTED_DATASETS: tuple[str, ...] = (
    "ruler",
    "loong",
    "long_bench_v2",
    "lara",
    "greek_bar_bench",
    "cuad",
    "clerk",
)

EMBEDDING_MODEL_ID = "BAAI/bge-m3"
EMBEDDING_MODEL_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
SNAPSHOT_PATH = (
    "D:/rag_work/hf_cache/models--BAAI--bge-m3/snapshots/"
    "5617a9f61b028005a4858fdac845db406aefb181"
)

CONDITION_ID = "LONGRAG-SD-4096"
CONSTRUCTION_POLICY = "single_document_if_within_budget"
TARGET_TOKEN_SIZE = 4096

SOURCE_ARCHITECTURE = "LongRAG"
SOURCE_ARCHITECTURE_COMMIT = "ba92f95"
ADAPTER_ID = "longrag_retrieval_adapter"
ADAPTER_VERSION = "0.2.0-source-corrected"

# The tokenizer is used offline from the pinned snapshot only. We import
# transformers lazily so that only this stage requires it.
def _load_tokenizer() -> Any:
    snapshot = Path(SNAPSHOT_PATH)
    if not snapshot.exists():
        raise FileNotFoundError(f"BGE-M3 snapshot not found: {snapshot}")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    from transformers import AutoTokenizer  # type: ignore[import-untyped]

    return AutoTokenizer.from_pretrained(
        str(snapshot),
        local_files_only=True,
    )


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
        .encode("utf-8")
        + b"\n"
    )


def atomic_write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(content)
    tmp.replace(path)


def write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    atomic_write_bytes(path, canonical_json_bytes(data))


def write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(
                json.dumps(
                    row, ensure_ascii=False, sort_keys=True,
                    separators=(",", ":"),
                )
            )
            fh.write("\n")
    tmp.replace(path)


def load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"Expected dict at {path}")
    return data


def build_segments_for_dataset(
    dataset_id: str,
    tokenizer: Any,
) -> tuple[tuple[LongRAGSourceSegment, ...], dict[str, int]]:
    """Stream canonical documents and build 1-segment-per-doc records."""
    segments: list[LongRAGSourceSegment] = []
    token_count_by_doc: dict[str, int] = {}
    for order, doc in enumerate(
        iter_canonical_documents(dataset_id)
    ):
        text = doc.text_normalized
        if not text.strip():
            raise RuntimeError(
                f"Document {doc.doc_id} has empty text_normalized."
            )
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        n_tokens = len(token_ids)
        if n_tokens == 0:
            raise RuntimeError(
                f"Document {doc.doc_id} tokenizes to zero tokens."
            )
        token_count_by_doc[doc.doc_id] = n_tokens
        segments.append(
            LongRAGSourceSegment(
                dataset_id=dataset_id,
                doc_id=doc.doc_id,
                parent_id=f"{doc.doc_id}:p0",
                chunk_id=f"seg-{doc.doc_id}",
                document_order=order,
                parent_order=0,
                chunk_order=0,
                text=text,
                token_count=n_tokens,
                metadata={"source": "canonical_document"},
            )
        )
    if not segments:
        raise RuntimeError(f"No segments produced for {dataset_id}.")
    return tuple(segments), token_count_by_doc


def split_oversized_segment(
    segment: LongRAGSourceSegment,
    target_token_size: int,
    tokenizer: Any,
    next_unit_order: int,
) -> list[LongUnit]:
    """Split one segment into fixed-size windows (source-corrected pattern)."""
    tokens = tokenizer.encode(segment.text, add_special_tokens=False)
    units: list[LongUnit] = []
    for part_index, start in enumerate(
        range(0, len(tokens), target_token_size)
    ):
        sub_tokens = tokens[start : start + target_token_size]
        sub_text = tokenizer.decode(sub_tokens, skip_special_tokens=True)
        if not sub_text.strip():
            continue
        sub_chunk_id = f"{segment.chunk_id}-part-{part_index}"
        unit_id = build_long_unit_id(
            dataset_id=segment.dataset_id,
            construction_policy=CONSTRUCTION_POLICY,  # type: ignore[arg-type]
            target_token_size=target_token_size,
            ordered_chunk_ids=(sub_chunk_id,),
        )
        units.append(
            LongUnit(
                long_unit_id=unit_id,
                dataset_id=segment.dataset_id,
                construction_policy=CONSTRUCTION_POLICY,  # type: ignore[arg-type]
                unit_order=next_unit_order + len(units),
                text=sub_text,
                token_count=len(sub_tokens),
                member_doc_ids=(segment.doc_id,),
                member_parent_ids=(segment.parent_id,),
                member_chunk_ids=(sub_chunk_id,),
                metadata={
                    "split": True,
                    "part_index": part_index,
                    "source_segment_chunk_id": segment.chunk_id,
                },
            )
        )
    return units


def construct_units(
    segments: tuple[LongRAGSourceSegment, ...],
    tokenizer: Any,
) -> tuple[LongUnit, ...]:
    """Adjacent-document packing with oversize split (faithful LongRAG).

    Faithful to the LongRAG paper: K consecutive documents are grouped
    into a long retrieval unit of ~target_token_size tokens. When a
    single document exceeds the target budget (ruler, loong,
    long_bench_v2), it is split into deterministic target-size windows.
    """
    units: list[LongUnit] = []
    current: list[LongRAGSourceSegment] = []

    def flush() -> None:
        if not current:
            return
        combined = "\n\n".join(s.text for s in current)
        chunk_ids = tuple(s.chunk_id for s in current)
        unit_id = build_long_unit_id(
            dataset_id=current[0].dataset_id,
            construction_policy=CONSTRUCTION_POLICY,  # type: ignore[arg-type]
            target_token_size=TARGET_TOKEN_SIZE,
            ordered_chunk_ids=chunk_ids,
        )
        rendered = len(tokenizer.encode(combined, add_special_tokens=False))
        units.append(
            LongUnit(
                long_unit_id=unit_id,
                dataset_id=current[0].dataset_id,
                construction_policy=CONSTRUCTION_POLICY,  # type: ignore[arg-type]
                unit_order=len(units),
                text=combined,
                token_count=rendered,
                member_doc_ids=tuple(
                    dict.fromkeys(s.doc_id for s in current)
                ),
                member_parent_ids=tuple(
                    dict.fromkeys(s.parent_id for s in current)
                ),
                member_chunk_ids=chunk_ids,
                metadata={"source_segment_count": len(current)},
            )
        )
        current.clear()

    for segment in segments:
        if segment.token_count > TARGET_TOKEN_SIZE:
            flush()
            units.extend(
                split_oversized_segment(
                    segment,
                    TARGET_TOKEN_SIZE,
                    tokenizer,
                    next_unit_order=len(units),
                )
            )
            continue

        if current:
            candidate_text = (
                "\n\n".join(s.text for s in current)
                + "\n\n"
                + segment.text
            )
            candidate_tokens = len(
                tokenizer.encode(candidate_text, add_special_tokens=False)
            )
            if candidate_tokens > TARGET_TOKEN_SIZE:
                flush()

        current.append(segment)

    flush()
    return tuple(units)


def build_summary_payload(
    dataset_id: str,
    segments: tuple[LongRAGSourceSegment, ...],
    units: tuple[LongUnit, ...],
) -> dict[str, Any]:
    """Compute summary with split-aware source coverage.

    A source segment that was split into multiple units contributes its
    original chunk_id through the per-unit metadata key
    ``source_segment_chunk_id``; unsplit segments contribute their own
    chunk_id directly. This preserves complete-source-coverage semantics
    under the oversize-split policy without mutating source identity.
    """
    src_doc_ids = {s.doc_id for s in segments}
    src_parent_ids = {s.parent_id for s in segments}
    src_chunk_ids = {s.chunk_id for s in segments}

    unit_doc_ids: set[str] = set()
    unit_parent_ids: set[str] = set()
    unit_chunk_ids: set[str] = set()
    for u in units:
        unit_doc_ids.update(u.member_doc_ids)
        unit_parent_ids.update(u.member_parent_ids)
        orig = u.metadata.get("source_segment_chunk_id")
        if orig is not None:
            unit_chunk_ids.add(str(orig))
        else:
            unit_chunk_ids.update(u.member_chunk_ids)

    token_counts = [u.token_count for u in units]
    min_tokens = min(token_counts) if token_counts else 0
    max_tokens = max(token_counts) if token_counts else 0
    avg_tokens = (
        sum(token_counts) / len(token_counts) if token_counts else 0.0
    )

    return {
        "dataset_id": dataset_id,
        "construction_policy": CONSTRUCTION_POLICY,
        "target_token_size": TARGET_TOKEN_SIZE,
        "source_segment_count": len(segments),
        "source_document_count": len(src_doc_ids),
        "source_parent_count": len(src_parent_ids),
        "long_unit_count": len(units),
        "minimum_unit_tokens": min_tokens,
        "maximum_unit_tokens": max_tokens,
        "average_unit_tokens": avg_tokens,
        "complete_document_coverage": unit_doc_ids == src_doc_ids,
        "complete_parent_coverage": unit_parent_ids == src_parent_ids,
        "complete_chunk_coverage": unit_chunk_ids == src_chunk_ids,
    }


def is_dataset_complete(ds_dir: Path) -> bool:
    manifest = ds_dir / "units-v1" / "manifest.json"
    if not manifest.exists():
        return False
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return data.get("status") == "SUCCEEDED"


def run_dataset(
    dataset_id: str,
    tokenizer: Any,
    *,
    force: bool,
) -> dict[str, Any]:
    ds_root = OUTPUT_ROOT / dataset_id
    stage_dir = ds_root / "units-v1"
    manifest_path = stage_dir / "manifest.json"

    if not force and is_dataset_complete(ds_root):
        print(f"[skip] {dataset_id} already SUCCEEDED")
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        return data

    t0 = time.perf_counter()
    print(f"[{dataset_id}] building segments ...")
    segments, _ = build_segments_for_dataset(dataset_id, tokenizer)
    print(
        f"[{dataset_id}] segments={len(segments)} "
        f"max_tokens={max(s.token_count for s in segments)}"
    )

    print(f"[{dataset_id}] constructing units ...")
    units = construct_units(segments, tokenizer)
    print(f"[{dataset_id}] units={len(units)}")

    max_unit_tokens = max(u.token_count for u in units)
    if max_unit_tokens > TARGET_TOKEN_SIZE:
        raise RuntimeError(
            f"{dataset_id}: max unit tokens {max_unit_tokens} "
            f"exceeds {TARGET_TOKEN_SIZE}"
        )

    summary = build_summary_payload(dataset_id, segments, units)
    if not summary["complete_document_coverage"]:
        raise RuntimeError(
            f"{dataset_id}: incomplete document coverage."
        )
    if not summary["complete_chunk_coverage"]:
        raise RuntimeError(
            f"{dataset_id}: incomplete chunk coverage."
        )

    stage_dir.mkdir(parents=True, exist_ok=True)
    units_path = stage_dir / "units.jsonl"
    summary_path = stage_dir / "summary.json"

    print(f"[{dataset_id}] writing units.jsonl ...")
    write_jsonl_atomic(
        units_path,
        [u.model_dump(mode="json") for u in units],
    )
    write_json_atomic(summary_path, summary)

    units_sha = sha256_file(units_path)
    summary_sha = sha256_file(summary_path)

    wall = time.perf_counter() - t0
    manifest: dict[str, Any] = {
        "artifact_type": "longrag_full_track_unit_materialization_manifest",
        "artifact_version": "1.0",
        "execution_id": f"longrag-l2-units-{dataset_id}-v1",
        "status": "SUCCEEDED",
        "dataset_id": dataset_id,
        "condition_id": CONDITION_ID,
        "construction_policy": CONSTRUCTION_POLICY,
        "target_token_size": TARGET_TOKEN_SIZE,
        "tokenizer_id": EMBEDDING_MODEL_ID,
        "tokenizer_revision": EMBEDDING_MODEL_REVISION,
        "source_architecture": SOURCE_ARCHITECTURE,
        "source_architecture_commit": SOURCE_ARCHITECTURE_COMMIT,
        "adapter_id": ADAPTER_ID,
        "adapter_version": ADAPTER_VERSION,
        "source_segment_count": len(segments),
        "long_unit_count": len(units),
        "max_unit_tokens": max_unit_tokens,
        "min_unit_tokens": min(u.token_count for u in units),
        "mean_unit_tokens": summary["average_unit_tokens"],
        "split_unit_count": sum(
            1 for u in units if u.metadata.get("split")
        ),
        "documents": summary["source_document_count"],
        "wall_clock_seconds": round(wall, 3),
        "units_jsonl_sha256": units_sha,
        "summary_json_sha256": summary_sha,
        "complete_document_coverage": summary[
            "complete_document_coverage"
        ],
        "complete_chunk_coverage": summary[
            "complete_chunk_coverage"
        ],
        "next_gate": "longrag_full_track_embeddings_prepare",
    }
    write_json_atomic(manifest_path, manifest)
    write_json_atomic(
        stage_dir / "checksums.json",
        {
            "files": {
                "units.jsonl": units_sha,
                "summary.json": summary_sha,
            }
        },
    )
    print(f"[{dataset_id}] DONE in {wall:.2f}s")
    return manifest


def run(*, force: bool, datasets: tuple[str, ...]) -> int:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    print("[L2] loading BGE-M3 tokenizer (offline snapshot) ...")
    tokenizer = _load_tokenizer()
    print(f"[L2] tokenizer ready: {EMBEDDING_MODEL_ID}")

    results: dict[str, Any] = {}
    for ds in datasets:
        try:
            results[ds] = run_dataset(ds, tokenizer, force=force)
        except Exception as exc:
            print(f"[{ds}] FAILED: {exc}")
            results[ds] = {
                "status": "FAILED",
                "error": str(exc),
                "dataset_id": ds,
            }

    root_manifest = {
        "artifact_type": "longrag_full_track_unit_materialization_manifest",
        "artifact_version": "1.0",
        "execution_id": "longrag-l2-all-datasets-v1",
        "status": "SUCCEEDED"
        if all(
            r.get("status") == "SUCCEEDED" for r in results.values()
        )
        else "PARTIAL",
        "datasets": {
            ds: {
                "status": results[ds].get("status"),
                "long_unit_count": results[ds].get("long_unit_count"),
                "documents": results[ds].get("documents"),
                "wall_clock_seconds": results[ds].get(
                    "wall_clock_seconds"
                ),
            }
            for ds in datasets
        },
    }
    write_json_atomic(OUTPUT_ROOT / "_l2_root_manifest.json", root_manifest)
    print()
    print("L2 SUMMARY")
    for ds in datasets:
        r = results[ds]
        print(
            f"  {ds:18s} "
            f"status={r.get('status')} "
            f"units={r.get('long_unit_count')} "
            f"docs={r.get('documents')} "
            f"secs={r.get('wall_clock_seconds')}"
        )
    return 0 if root_manifest["status"] == "SUCCEEDED" else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force", action="store_true",
        help="Recompute even if manifest is SUCCEEDED.",
    )
    parser.add_argument(
        "--datasets", nargs="*", default=list(EXPECTED_DATASETS),
        choices=list(EXPECTED_DATASETS),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return run(force=args.force, datasets=tuple(args.datasets))


if __name__ == "__main__":
    raise SystemExit(main())