"""Generate deterministic, resumable LongRAG long-unit embedding shards.

CPU-only BGE-M3 inference. Does not ingest Qdrant, does not perform retrieval.
Resume-safe: complete shards are skipped on rerun.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from FlagEmbedding import BGEM3FlagModel
from transformers import AutoTokenizer

from rag_system.artifacts import (
    ArtifactComponentReference,
    build_file_sidecar_manifest,
    build_input_reference,
    get_git_revision,
    sha256_fingerprint,
    write_json_model_atomic,
)
from rag_system.baselines import (
    LongUnitEmbeddingCatalog,
    LongUnitEmbeddingRecord,
    LongUnitEmbeddingShardHeader,
)
from rag_system.embedding import EmbeddingShardCatalogEntry

MODEL_REPORT_PATH = Path("outputs/bge_m3_model_download.json")
UNITS_ROOT = Path("outputs/longrag_clerk_source_corrected_pilot")
OUTPUT_ROOT = Path("outputs/longrag_clerk_bge_m3_embeddings")

MODEL_ID = "BAAI/bge-m3"
MODEL_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
TOKENIZER_ID = "BAAI/bge-m3"
DENSE_DIMENSION = 1024
DENSE_PRECISION = "float32"
DENSE_NORMALIZED = True

ADAPTER_ID = "longrag_retrieval_adapter"
ADAPTER_VERSION = "0.2.0-source-corrected"
SOURCE_ARCHITECTURE = "LongRAG"
SOURCE_ARCHITECTURE_COMMIT = "ba92f95"
DATASET_ID = "clerk"

CONDITIONS: dict[str, dict[str, int]] = {
    "LONGRAG-SD-4096": {
        "expected_units": 2756,
        "max_length": 1024,
        "batch_size": 4,
        "shard_size": 256,
    },
    "LONGRAG-ADJ-2048": {
        "expected_units": 633,
        "max_length": 2050,
        "batch_size": 1,
        "shard_size": 64,
    },
    "LONGRAG-ADJ-4096": {
        "expected_units": 293,
        "max_length": 4098,
        "batch_size": 1,
        "shard_size": 32,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build resumable LongRAG embedding shards."
    )
    parser.add_argument(
        "--condition",
        required=True,
        choices=list(CONDITIONS.keys()),
        help="Target LongRAG condition.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore resume cache and regenerate all shards.",
    )
    return parser.parse_args()


def canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True).encode(
            "utf-8"
        )
        + b"\n"
    )


def atomic_write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(content)
    tmp.replace(path)


def atomic_write_jsonl(
    path: Path,
    records: list[LongUnitEmbeddingRecord],
) -> None:
    """Atomically write embedding records as JSONL."""
    content = b"".join(
        canonical_json_bytes(record.model_dump(mode="json"))
        for record in records
    )
    atomic_write_bytes(path, content)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def sha256_value(value: object) -> str:
    """Calculate one deterministic value fingerprint."""
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def load_units(
    condition_id: str,
) -> tuple[Path, list[dict[str, Any]]]:
    """Load one validated source-unit file."""
    path = UNITS_ROOT / condition_id / "units.jsonl"

    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    expected = CONDITIONS[condition_id]["expected_units"]

    if len(rows) != expected:
        raise RuntimeError(
            f"{condition_id}: expected {expected} units, received {len(rows)}."
        )

    unit_ids = [str(row["long_unit_id"]) for row in rows]

    if len(unit_ids) != len(set(unit_ids)):
        raise RuntimeError("Source units contain duplicate identities.")

    return path, rows


def dense_matrix(
    output: dict[str, Any],
    *,
    expected_rows: int,
) -> np.ndarray:
    """Validate one dense embedding matrix."""
    dense = output.get("dense_vecs")

    if dense is None:
        raise RuntimeError("BGE-M3 output has no dense_vecs.")

    matrix = np.asarray(dense, dtype=np.float32)

    if matrix.shape != (expected_rows, DENSE_DIMENSION):
        raise RuntimeError(
            f"Unexpected dense embedding shape: {matrix.shape}."
        )

    if not np.isfinite(matrix).all():
        raise RuntimeError("Dense matrix contains non-finite values.")

    norms = np.linalg.norm(matrix, axis=1)

    if not np.allclose(norms, 1.0, atol=1e-3):
        raise RuntimeError(
            f"Dense vectors not L2-normalized: min={float(norms.min())}, "
            f"max={float(norms.max())}."
        )

    return matrix


def embedding_config_hash(
    condition_id: str,
    max_length: int,
    batch_size: int,
) -> str:
    return sha256_fingerprint(
        {
            "condition_id": condition_id,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "tokenizer_id": TOKENIZER_ID,
            "max_length": max_length,
            "batch_size": batch_size,
            "dense_dimension": DENSE_DIMENSION,
            "dense_precision": DENSE_PRECISION,
            "dense_normalized": DENSE_NORMALIZED,
        }
    )


def shard_paths(cond_root: Path, index: int) -> tuple[Path, Path, Path]:
    return (
        cond_root / f"shard-{index:04d}.jsonl",
        cond_root / f"shard-{index:04d}.header.json",
        cond_root / f"shard-{index:04d}.manifest.json",
    )


def count_lines(path: Path) -> int:
    with path.open("rb") as f:
        return sum(1 for _ in f)


def shard_is_complete(
    records_path: Path,
    header_path: Path,
    manifest_path: Path,
    *,
    expected_records: int,
    expected_config_hash: str,
    expected_source_sha: str,
) -> bool:
    if not (
        records_path.is_file()
        and header_path.is_file()
        and manifest_path.is_file()
    ):
        return False

    try:
        header = json.loads(header_path.read_text(encoding="utf-8"))
    except Exception:
        return False

    if header.get("embedding_config_hash") != expected_config_hash:
        return False
    if header.get("source_units_sha256") != expected_source_sha:
        return False
    if header.get("paper_faithful_result") is not False:
        return False

    try:
        stored = count_lines(records_path)
    except Exception:
        return False

    return stored == expected_records


def build_catalog_entry(
    *,
    cond_root: Path,
    shard_index: int,
    source_start_index: int,
    record_count: int,
) -> EmbeddingShardCatalogEntry:
    """Build one catalog entry for a fully written shard."""
    records_path, header_path, manifest_path = shard_paths(
        cond_root, shard_index
    )

    if not (
        records_path.is_file()
        and header_path.is_file()
        and manifest_path.is_file()
    ):
        raise RuntimeError(
            f"Shard {shard_index}: missing artifacts under {cond_root}."
        )

    return EmbeddingShardCatalogEntry(
        shard_id=f"{cond_root.name}-shard-{shard_index:04d}",
        shard_index=shard_index,
        shard_root=str(cond_root),
        header_path=str(header_path),
        records_path=str(records_path),
        manifest_path=str(manifest_path),
        header_sha256=sha256_file(header_path),
        records_sha256=sha256_file(records_path),
        manifest_sha256=sha256_file(manifest_path),
        record_count=record_count,
        source_start_index=source_start_index,
        source_end_index=source_start_index + record_count,
    )


def process_shard(
    *,
    model: BGEM3FlagModel,
    tokenizer: Any,
    condition_id: str,
    cond_root: Path,
    shard_index: int,
    units: list[dict[str, Any]],
    max_length: int,
    batch_size: int,
    shard_size: int,
    special_token_count: int,
    config_hash: str,
    source_units_path: Path,
    source_units_sha256: str,
) -> dict[str, Any]:
    records_path, header_path, manifest_path = shard_paths(
        cond_root, shard_index
    )

    texts = [str(u["text"]) for u in units]
    rendered = [
        len(tokenizer.encode(t, add_special_tokens=False)) for t in texts
    ]
    actual = [
        len(tokenizer.encode(t, add_special_tokens=True)) for t in texts
    ]

    for i, (r, a) in enumerate(zip(rendered, actual, strict=True)):
        if r + special_token_count != a:
            raise RuntimeError(
                f"{condition_id} shard {shard_index} unit {i}: "
                f"rendered({r}) + special({special_token_count}) != actual({a})"
            )
        if a > max_length:
            raise RuntimeError(
                f"{condition_id} shard {shard_index} unit {i}: "
                f"actual({a}) > max_length({max_length}) — would truncate"
            )

    started = time.perf_counter()
    output = model.encode(
        texts,
        batch_size=batch_size,
        max_length=max_length,
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False,
    )
    elapsed = time.perf_counter() - started

    matrix = dense_matrix(output, expected_rows=len(units))

    records: list[LongUnitEmbeddingRecord] = []
    for _i, (unit, vec, r_count, a_count) in enumerate(
        zip(units, matrix, rendered, actual, strict=True)
    ):
        unit_sha = sha256_value(unit)
        records.append(
            LongUnitEmbeddingRecord(
                long_unit_id=str(unit["long_unit_id"]),
                condition_id=condition_id,
                dataset_id=str(unit["dataset_id"]),
                construction_policy=str(unit["construction_policy"]),
                target_token_size=int(unit["target_token_size"]),
                maximum_model_input_tokens=max_length,
                unit_order=int(unit["unit_order"]),
                rendered_token_count=r_count,
                special_token_count=special_token_count,
                actual_model_token_count=a_count,
                was_truncated=False,
                member_doc_ids=tuple(str(x) for x in unit["member_doc_ids"]),
                member_parent_ids=tuple(
                    str(x) for x in unit["member_parent_ids"]
                ),
                member_chunk_ids=tuple(
                    str(x) for x in unit["member_chunk_ids"]
                ),
                model_id=MODEL_ID,
                model_revision=MODEL_REVISION,
                tokenizer_id=TOKENIZER_ID,
                embedding_config_hash=config_hash,
                dense_vector=tuple(float(x) for x in vec),
                dense_dimension=DENSE_DIMENSION,
                dense_precision=DENSE_PRECISION,
                dense_normalized=DENSE_NORMALIZED,
                source_unit_sha256=unit_sha,
                source_units_file_sha256=source_units_sha256,
                adapter_id=ADAPTER_ID,
                adapter_version=ADAPTER_VERSION,
                source_architecture=SOURCE_ARCHITECTURE,
                source_architecture_commit=SOURCE_ARCHITECTURE_COMMIT,
                interface_validation=True,
                paper_faithful_result=False,
            )
        )

    ids = [r.long_unit_id for r in records]
    if len(ids) != len(set(ids)):
        raise RuntimeError(
            f"{condition_id} shard {shard_index}: duplicate long_unit_id."
        )

    atomic_write_jsonl(records_path, records)

    shard_id = f"{condition_id}-shard-{shard_index:04d}"
    header = LongUnitEmbeddingShardHeader(
        shard_id=shard_id,
        shard_index=shard_index,
        dataset_id=DATASET_ID,
        condition_id=condition_id,
        model_id=MODEL_ID,
        model_revision=MODEL_REVISION,
        tokenizer_id=TOKENIZER_ID,
        embedding_config_hash=config_hash,
        dense_dimension=DENSE_DIMENSION,
        dense_precision=DENSE_PRECISION,
        dense_normalized=DENSE_NORMALIZED,
        maximum_model_input_tokens=max_length,
        special_token_count=special_token_count,
        maximum_records=shard_size,
        adapter_id=ADAPTER_ID,
        adapter_version=ADAPTER_VERSION,
        source_architecture_commit=SOURCE_ARCHITECTURE_COMMIT,
        source_units_path=str(source_units_path),
        source_units_sha256=source_units_sha256,
        interface_validation=True,
        paper_faithful_result=False,
    )
    write_json_model_atomic(header_path, header, overwrite=True)

    component_config = {
        "condition_id": condition_id,
        "shard_index": shard_index,
        "max_length": max_length,
        "batch_size": batch_size,
        "shard_size": shard_size,
        "embedding_config_hash": config_hash,
    }
    component = ArtifactComponentReference(
        component_type="longrag_embedding_shard",
        component_id=f"{ADAPTER_ID}-{condition_id}",
        component_version=ADAPTER_VERSION,
        configuration_hash=sha256_fingerprint(component_config),
        configuration=component_config,
    )
    input_ref = build_input_reference(
        artifact_type="longrag_source_units",
        path=source_units_path,
        record_count=len(units),
    )
    manifest = build_file_sidecar_manifest(
        artifact_type="longrag_embedding_shard",
        artifact_path=records_path,
        record_count=len(records),
        component=component,
        dataset_id=DATASET_ID,
        inputs=(input_ref,),
        code_revision=get_git_revision(),
        execution_profile="local_cpu",
        metadata={
            "condition_id": condition_id,
            "shard_index": shard_index,
            "embedding_seconds": elapsed,
            "paper_faithful_result": False,
        },
    )
    write_json_model_atomic(manifest_path, manifest, overwrite=True)

    return {
        "shard_index": shard_index,
        "status": "written",
        "records": len(records),
        "seconds": round(elapsed, 3),
    }


def main() -> None:
    args = parse_args()
    condition_id = args.condition
    cfg = CONDITIONS[condition_id]
    max_length = int(cfg["max_length"])
    batch_size = int(cfg["batch_size"])
    shard_size = int(cfg["shard_size"])

    cond_root = OUTPUT_ROOT / condition_id
    cond_root.mkdir(parents=True, exist_ok=True)

    model_report = json.loads(
        MODEL_REPORT_PATH.read_text(encoding="utf-8")
    )
    snapshot_path = Path(model_report["snapshot_path"])

    print(f"[INFO] {condition_id}")
    print(f"[INFO] units root: {UNITS_ROOT}")
    print(f"[INFO] output root: {cond_root}")
    print(f"[INFO] snapshot: {snapshot_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        str(snapshot_path), local_files_only=True
    )
    special_token_count = int(tokenizer.num_special_tokens_to_add(pair=False))
    print(f"[INFO] special_token_count={special_token_count}")

    units_path, units = load_units(condition_id)
    source_units_sha256 = sha256_file(units_path)
    print(f"[INFO] loaded {len(units)} units")
    print(f"[INFO] source_units_sha256={source_units_sha256}")

    config_hash = embedding_config_hash(
        condition_id, max_length, batch_size
    )
    print(f"[INFO] embedding_config_hash={config_hash}")
    print(f"[INFO] max_length={max_length} batch_size={batch_size} shard_size={shard_size}")

    shards: list[list[dict[str, Any]]] = [
        units[i : i + shard_size]
        for i in range(0, len(units), shard_size)
    ]
    print(f"[INFO] shard count: {len(shards)}")

    pending: list[tuple[int, list[dict[str, Any]]]] = []
    skipped: list[int] = []

    for idx, shard_units in enumerate(shards):
        records_path, header_path, manifest_path = shard_paths(cond_root, idx)
        if not args.force and shard_is_complete(
            records_path,
            header_path,
            manifest_path,
            expected_records=len(shard_units),
            expected_config_hash=config_hash,
            expected_source_sha=source_units_sha256,
        ):
            print(
                f"[SKIP] shard {idx:04d} records={len(shard_units)}"
            )
            skipped.append(idx)
            continue
        pending.append((idx, shard_units))

    if pending:
        gc.collect()
        print(f"[INFO] loading BGEM3FlagModel (CPU) from {snapshot_path}")
        model = BGEM3FlagModel(
            str(snapshot_path), use_fp16=False, device="cpu"
        )

        for idx, shard_units in pending:
            t0 = time.perf_counter()
            info = process_shard(
                model=model,
                tokenizer=tokenizer,
                condition_id=condition_id,
                cond_root=cond_root,
                shard_index=idx,
                units=shard_units,
                max_length=max_length,
                batch_size=batch_size,
                shard_size=shard_size,
                special_token_count=special_token_count,
                config_hash=config_hash,
                source_units_path=units_path,
                source_units_sha256=source_units_sha256,
            )
            print(
                f"[WRITE] shard {idx:04d} "
                f"records={info['records']} "
                f"seconds={info['seconds']} "
                f"elapsed_total={round(time.perf_counter() - t0, 3)}"
            )

        del model
        gc.collect()
    else:
        print("[INFO] all shards already complete")

    entries: list[EmbeddingShardCatalogEntry] = []
    total_records = 0
    doc_ids: set[str] = set()

    for idx, shard_units in enumerate(shards):
        records_path, header_path, _manifest_path = shard_paths(
            cond_root, idx
        )
        if not records_path.is_file() or not header_path.is_file():
            raise RuntimeError(
                f"Missing shard artifacts for index {idx}."
            )
        line_count = count_lines(records_path)
        if line_count != len(shard_units):
            raise RuntimeError(
                f"Shard {idx}: expected {len(shard_units)} records, "
                f"got {line_count}."
            )
        source_start_index = idx * shard_size
        entries.append(
            build_catalog_entry(
                cond_root=cond_root,
                shard_index=idx,
                source_start_index=source_start_index,
                record_count=line_count,
            )
        )
        total_records += line_count
        for u in shard_units:
            for doc_id in u["member_doc_ids"]:
                doc_ids.add(str(doc_id))

    first_unit = units[0]
    run_id = sha256_fingerprint(
        {
            "condition_id": condition_id,
            "embedding_config_hash": config_hash,
            "source_units_sha256": source_units_sha256,
        }
    )

    catalog = LongUnitEmbeddingCatalog(
        run_id=run_id,
        dataset_id=DATASET_ID,
        condition_id=condition_id,
        construction_policy=str(first_unit["construction_policy"]),
        target_token_size=int(first_unit["target_token_size"]),
        maximum_model_input_tokens=max_length,
        special_token_count=special_token_count,
        model_id=MODEL_ID,
        model_revision=MODEL_REVISION,
        tokenizer_id=TOKENIZER_ID,
        embedding_config_hash=config_hash,
        dense_dimension=DENSE_DIMENSION,
        dense_precision=DENSE_PRECISION,
        dense_normalized=DENSE_NORMALIZED,
        source_units_path=str(units_path),
        source_units_sha256=source_units_sha256,
        shard_count=len(entries),
        record_count=total_records,
        document_count=len(doc_ids),
        shards=tuple(entries),
        adapter_id=ADAPTER_ID,
        adapter_version=ADAPTER_VERSION,
        source_architecture=SOURCE_ARCHITECTURE,
        source_architecture_commit=SOURCE_ARCHITECTURE_COMMIT,
        code_revision=get_git_revision(),
        execution_profile="local_cpu",
        interface_validation=True,
        paper_faithful_result=False,
    )

    catalog_path = cond_root / "embedding_catalog.json"
    write_json_model_atomic(catalog_path, catalog, overwrite=True)

    print(f"[OK] catalog: {catalog_path}")
    print(
        f"     shards={len(entries)} records={total_records} "
        f"documents={len(doc_ids)} skipped={len(skipped)}"
    )


if __name__ == "__main__":
    main()
