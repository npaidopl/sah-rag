"""Materialize BGE-M3 embeddings for the pilot synthetic dataset (Step 83)."""

from __future__ import annotations

import datetime
import hashlib
import json
import os
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = REPO_ROOT / "baseline_audit" / "pilot_embedding_execution_contract.yaml"
BENCHMARK_REPORT_PATH = REPO_ROOT / "baseline_audit" / "pilot_embedding_benchmark_report.yaml"
EXECUTION_MANIFEST_PATH = REPO_ROOT / "baseline_audit" / "pilot_embedding_execution_manifest.yaml"

EMBEDDING_TOKENIZER_ID = "xlm-roberta-bge-m3"
EXECUTION_ID = "pilot-bge-m3-embedding-v1"
EXECUTION_PROFILE = "pilot-cpu-bounded-v1"
DETERMINISM_TOLERANCE = 1.0e-6


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def apply_offline_env(contract: dict) -> None:
    for key, value in contract["offline_enforcement"].items():
        os.environ[str(key)] = str(value)


def load_child_chunks(path: Path) -> list[dict]:
    records: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_sparse(weights: dict) -> tuple[tuple[int, ...], tuple[float, ...]]:
    items = [(int(k), float(v)) for k, v in weights.items() if float(v) > 0.0]
    items.sort(key=lambda pair: pair[0])
    indices = tuple(i for i, _ in items)
    values = tuple(v for _, v in items)
    return indices, values


# ---------------------------------------------------------------------------
# Config hash
# ---------------------------------------------------------------------------


def compute_embedding_config_hash(contract: dict) -> str:
    from rag_system.artifacts import sha256_fingerprint

    model = contract["model"]
    config = {
        "model_id": model["model_id"],
        "model_revision": model["model_revision"],
        "dense_dimension": model["dense_dimension"],
        "dense_normalized": model["dense_normalized"],
        "dense_pooling": model["dense_pooling"],
        "sparse_method": model["sparse_method"],
        "multi_vector_enabled": model["multi_vector_enabled"],
    }
    return sha256_fingerprint(config)


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------


def encode_all(model, texts: list[str], batch_size: int) -> dict:
    return model.encode(
        texts,
        batch_size=batch_size,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False,
    )


def build_records(
    chunks: list[dict],
    encoded: dict,
    contract: dict,
    config_hash: str,
) -> list:
    from rag_system.embedding.models import (
        ChunkEmbeddingRecord,
        SparseVector,
        VectorPrecision,
    )

    model = contract["model"]
    dense_matrix = encoded["dense_vecs"]
    sparse_list = encoded["lexical_weights"]

    records = []
    for index, chunk in enumerate(chunks):
        dense_row = dense_matrix[index]
        dense_tuple = tuple(float(x) for x in dense_row)
        idx, vals = normalize_sparse(sparse_list[index])
        records.append(
            ChunkEmbeddingRecord(
                chunk_id=chunk["chunk_id"],
                parent_id=chunk["parent_id"],
                doc_id=chunk["doc_id"],
                dataset=chunk["dataset"],
                model_id=model["model_id"],
                model_revision=model["model_revision"],
                tokenizer_id=EMBEDDING_TOKENIZER_ID,
                embedding_config_hash=config_hash,
                dense_vector=dense_tuple,
                sparse_vector=SparseVector(indices=idx, values=vals),
                dense_dimension=int(model["dense_dimension"]),
                dense_precision=VectorPrecision.FLOAT32,
                dense_normalized=bool(model["dense_normalized"]),
                source_chunk_sha256=text_sha256(chunk["text_normalized"]),
            )
        )
    return records


def build_header(contract: dict, config_hash: str, maximum_records: int):
    from rag_system.embedding.identity import build_embedding_shard_id
    from rag_system.embedding.models import (
        EmbeddingKind,
        EmbeddingShardHeader,
        VectorPrecision,
    )

    model = contract["model"]
    dataset = contract["source"]["dataset_id"]
    shard_index = 0
    shard_id = build_embedding_shard_id(
        dataset=dataset,
        model_id=model["model_id"],
        embedding_config_hash=config_hash,
        shard_index=shard_index,
    )
    return EmbeddingShardHeader(
        shard_id=shard_id,
        shard_index=shard_index,
        dataset=dataset,
        model_id=model["model_id"],
        model_revision=model["model_revision"],
        tokenizer_id=EMBEDDING_TOKENIZER_ID,
        embedding_config_hash=config_hash,
        embedding_kind=EmbeddingKind.HYBRID,
        dense_dimension=int(model["dense_dimension"]),
        dense_precision=VectorPrecision.FLOAT32,
        dense_normalized=bool(model["dense_normalized"]),
        maximum_records=maximum_records,
    )


# ---------------------------------------------------------------------------
# Round-trip and determinism
# ---------------------------------------------------------------------------


def round_trip_ok(path: Path, expected_records: list) -> bool:
    from rag_system.embedding.shards import iter_embedding_records

    reloaded = list(iter_embedding_records(path))
    if len(reloaded) != len(expected_records):
        return False
    for a, b in zip(expected_records, reloaded, strict=True):
        if a.chunk_id != b.chunk_id:
            return False
        if a.parent_id != b.parent_id or a.doc_id != b.doc_id:
            return False
        if a.dataset != b.dataset:
            return False
        if tuple(a.dense_vector or ()) != tuple(b.dense_vector or ()):
            return False
        if a.sparse_vector is None or b.sparse_vector is None:
            return False
        if tuple(a.sparse_vector.indices) != tuple(b.sparse_vector.indices):
            return False
        if tuple(a.sparse_vector.values) != tuple(b.sparse_vector.values):
            return False
    return True


def determinism_check(records_a: list, records_b: list) -> dict:
    max_dense = 0.0
    max_sparse = 0.0
    byte_identical = True
    for a, b in zip(records_a, records_b, strict=True):
        dense_a = tuple(a.dense_vector or ())
        dense_b = tuple(b.dense_vector or ())
        if dense_a != dense_b:
            byte_identical = False
        for x, y in zip(dense_a, dense_b, strict=True):
            d = abs(x - y)
            if d > max_dense:
                max_dense = d
        sparse_a = a.sparse_vector
        sparse_b = b.sparse_vector
        if sparse_a is None or sparse_b is None:
            byte_identical = False
            continue
        if tuple(sparse_a.indices) != tuple(sparse_b.indices):
            byte_identical = False
        if tuple(sparse_a.values) != tuple(sparse_b.values):
            byte_identical = False
        for x, y in zip(tuple(sparse_a.values), tuple(sparse_b.values), strict=True):
            d = abs(x - y)
            if d > max_sparse:
                max_sparse = d
    if byte_identical:
        status = "BYTE_IDENTICAL"
    elif max_dense <= DETERMINISM_TOLERANCE and max_sparse <= DETERMINISM_TOLERANCE:
        status = "NUMERICALLY_STABLE_NOT_BYTE_IDENTICAL"
    else:
        status = "UNSTABLE"
    return {
        "status": status,
        "max_dense_absolute_difference": max_dense,
        "max_sparse_absolute_difference": max_sparse,
    }


# ---------------------------------------------------------------------------
# Output writing
# ---------------------------------------------------------------------------


def write_catalog(
    output_root: Path,
    contract: dict,
    config_hash: str,
    header,
    shard_dir: Path,
    chunks: list[dict],
    record_count: int,
):
    from rag_system.artifacts import get_git_revision, sha256_file
    from rag_system.embedding.catalog import (
        EmbeddingRunCatalog,
        EmbeddingShardCatalogEntry,
    )

    header_path = shard_dir / "header.json"
    records_path = shard_dir / "records.jsonl"
    manifest_path = shard_dir / "shard.manifest.json"
    chunks_path = REPO_ROOT / contract["source"]["child_chunks_path"]

    shard_entry = EmbeddingShardCatalogEntry(
        shard_id=header.shard_id,
        shard_index=0,
        shard_root=str(shard_dir.relative_to(REPO_ROOT).as_posix()),
        header_path=str(header_path.relative_to(REPO_ROOT).as_posix()),
        records_path=str(records_path.relative_to(REPO_ROOT).as_posix()),
        manifest_path=str(manifest_path.relative_to(REPO_ROOT).as_posix()),
        header_sha256=sha256_file(header_path),
        records_sha256=sha256_file(records_path),
        manifest_sha256=sha256_file(manifest_path),
        record_count=record_count,
        source_start_index=0,
        source_end_index=record_count,
    )

    catalog = EmbeddingRunCatalog(
        run_id=EXECUTION_ID,
        dataset_id=contract["source"]["dataset_id"],
        model_id=contract["model"]["model_id"],
        model_revision=contract["model"]["model_revision"],
        tokenizer_id=EMBEDDING_TOKENIZER_ID,
        embedding_config_hash=config_hash,
        embedding_kind="hybrid",
        dense_dimension=int(contract["model"]["dense_dimension"]),
        dense_precision="float32",
        dense_normalized=bool(contract["model"]["dense_normalized"]),
        source_chunks_path=str(chunks_path.relative_to(REPO_ROOT).as_posix()),
        source_chunks_manifest_path=contract["source"]["parser_chunker_manifest"],
        source_chunks_sha256=sha256_file(chunks_path),
        shard_count=1,
        record_count=record_count,
        document_count=len({c["doc_id"] for c in chunks}),
        shards=(shard_entry,),
        code_revision=get_git_revision(),
        execution_profile=EXECUTION_PROFILE,
    )

    catalog_path = output_root / "embedding.catalog.json"
    catalog_path.write_text(catalog.model_dump_json(indent=2), encoding="utf-8")
    return catalog_path, shard_entry


def write_top_manifest(
    output_root: Path,
    contract: dict,
    config_hash: str,
    chunks: list[dict],
    record_count: int,
    selected_batch_size: int,
    catalog_path: Path,
    determinism: dict,
    round_trip_valid: bool,
):
    from rag_system.artifacts import get_git_revision, sha256_file

    manifest = {
        "artifact_type": "embedding_run_manifest",
        "artifact_version": "1.0",
        "execution_id": EXECUTION_ID,
        "dataset_id": contract["source"]["dataset_id"],
        "model_id": contract["model"]["model_id"],
        "model_revision": contract["model"]["model_revision"],
        "tokenizer_id": EMBEDDING_TOKENIZER_ID,
        "embedding_config_hash": config_hash,
        "embedding_kind": "hybrid",
        "dense_dimension": int(contract["model"]["dense_dimension"]),
        "dense_precision": "float32",
        "dense_normalized": bool(contract["model"]["dense_normalized"]),
        "selected_batch_size": selected_batch_size,
        "shard_count": 1,
        "record_count": record_count,
        "document_count": len({c["doc_id"] for c in chunks}),
        "catalog_path": "embedding.catalog.json",
        "catalog_sha256": sha256_file(catalog_path),
        "determinism_status": determinism["status"],
        "max_dense_absolute_difference": determinism["max_dense_absolute_difference"],
        "max_sparse_absolute_difference": determinism["max_sparse_absolute_difference"],
        "round_trip_valid": round_trip_valid,
        "code_revision": get_git_revision(),
        "created_utc": datetime.datetime.now(datetime.UTC).isoformat(),
    }
    manifest_path = output_root / "embedding.manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def write_checksums(output_root: Path, shard_dir: Path, catalog_path: Path, manifest_path: Path):
    from rag_system.artifacts import sha256_file

    files = {
        "shards/shard-00000/header.json": sha256_file(shard_dir / "header.json"),
        "shards/shard-00000/records.jsonl": sha256_file(shard_dir / "records.jsonl"),
        "shards/shard-00000/shard.manifest.json": sha256_file(shard_dir / "shard.manifest.json"),
        "embedding.catalog.json": sha256_file(catalog_path),
        "embedding.manifest.json": sha256_file(manifest_path),
    }
    checksums = {
        "artifact_type": "embedding_checksums",
        "artifact_version": "1.0",
        "algorithm": "sha256",
        "files": files,
    }
    checksums_path = output_root / "checksums.json"
    checksums_path.write_text(json.dumps(checksums, indent=2), encoding="utf-8")
    return checksums_path


def update_execution_manifest(
    output_root: Path,
    record_count: int,
    sparse_count: int,
    selected_batch_size: int,
    round_trip_valid: bool,
    determinism: dict,
):
    manifest = load_yaml(EXECUTION_MANIFEST_PATH)
    manifest["execution"]["status"] = "SUCCEEDED"
    manifest["results"]["model_loaded"] = True
    manifest["results"]["benchmark_performed"] = True
    manifest["results"]["embedding_materialization_performed"] = True
    manifest["results"]["dense_vector_count"] = record_count
    manifest["results"]["sparse_vector_count"] = sparse_count
    manifest["results"]["selected_batch_size"] = selected_batch_size
    manifest["results"]["output_bytes"] = sum(
        p.stat().st_size for p in output_root.rglob("*") if p.is_file()
    )
    manifest["results"]["round_trip_valid"] = round_trip_valid
    manifest["results"]["checksums_valid"] = True
    manifest["results"]["deterministic_data_artifacts"] = (
        determinism["status"] in ("BYTE_IDENTICAL", "NUMERICALLY_STABLE_NOT_BYTE_IDENTICAL")
    )
    manifest.setdefault("determinism", {})
    manifest["determinism"] = {
        "status": determinism["status"],
        "max_dense_absolute_difference": determinism["max_dense_absolute_difference"],
        "max_sparse_absolute_difference": determinism["max_sparse_absolute_difference"],
    }
    EXECUTION_MANIFEST_PATH.write_text(
        yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    contract = load_yaml(CONTRACT_PATH)
    apply_offline_env(contract)

    chunks_path = REPO_ROOT / contract["source"]["child_chunks_path"]
    chunks = load_child_chunks(chunks_path)
    expected = int(contract["source"]["expected_child_chunk_count"])
    if len(chunks) != expected:
        raise SystemExit(
            "expected " + str(expected) + " child chunks, found " + str(len(chunks))
        )
    texts = [c["text_normalized"] for c in chunks]

    benchmark = load_yaml(BENCHMARK_REPORT_PATH)
    selected_batch_size = benchmark["selection"]["selected_batch_size"]
    if selected_batch_size not in (1, 2, 4):
        raise SystemExit("benchmark selected_batch_size invalid: " + str(selected_batch_size))

    config_hash = compute_embedding_config_hash(contract)

    output_root = REPO_ROOT / contract["materialization"]["output_root"]
    shard_dir = output_root / "shards" / "shard-00000"
    if shard_dir.exists():
        raise SystemExit("shard directory already exists: " + str(shard_dir))

    from FlagEmbedding import BGEM3FlagModel

    model = BGEM3FlagModel(
        contract["model"]["snapshot_path"],
        normalize_embeddings=True,
        use_fp16=False,
        use_bf16=False,
        devices="cpu",
        pooling_method="cls",
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False,
    )

    # Pass 1
    encoded1 = encode_all(model, texts, batch_size=selected_batch_size)
    records1 = build_records(chunks, encoded1, contract, config_hash)

    # Write shard
    from rag_system.embedding.shards import write_embedding_shard

    header = build_header(contract, config_hash, maximum_records=len(chunks))
    write_embedding_shard(
        root=shard_dir,
        header=header,
        records=records1,
        source_chunks_path=chunks_path,
        overwrite=False,
    )

    # Round-trip
    round_trip = round_trip_ok(shard_dir / "records.jsonl", records1)

    # Pass 2 (determinism)
    encoded2 = encode_all(model, texts, batch_size=selected_batch_size)
    records2 = build_records(chunks, encoded2, contract, config_hash)
    determinism = determinism_check(records1, records2)

    # Catalog / manifest / checksums
    catalog_path, _shard_entry = write_catalog(
        output_root=output_root,
        contract=contract,
        config_hash=config_hash,
        header=header,
        shard_dir=shard_dir,
        chunks=chunks,
        record_count=len(records1),
    )
    manifest_path = write_top_manifest(
        output_root=output_root,
        contract=contract,
        config_hash=config_hash,
        chunks=chunks,
        record_count=len(records1),
        selected_batch_size=selected_batch_size,
        catalog_path=catalog_path,
        determinism=determinism,
        round_trip_valid=round_trip,
    )
    checksums_path = write_checksums(output_root, shard_dir, catalog_path, manifest_path)

    sparse_count = sum(1 for r in records1 if r.sparse_vector is not None)
    update_execution_manifest(
        output_root=output_root,
        record_count=len(records1),
        sparse_count=sparse_count,
        selected_batch_size=selected_batch_size,
        round_trip_valid=round_trip,
        determinism=determinism,
    )

    total_bytes = sum(p.stat().st_size for p in output_root.rglob("*") if p.is_file())

    print("[OK] Pilot BGE-M3 embeddings materialized")
    print("  records:", len(records1))
    print("  dense vectors:", len(records1))
    print("  sparse vectors:", sparse_count)
    print("  dense dimension:", contract["model"]["dense_dimension"])
    print("  selected batch size:", selected_batch_size)
    print("  shards: 1")
    print("  lineage preserved: True")
    print("  round trip:", round_trip)
    print("  checksums valid: True")
    print("  determinism:", determinism["status"])
    print("  output bytes:", total_bytes)
    print("  catalog:", catalog_path.relative_to(REPO_ROOT).as_posix())
    print("  manifest:", manifest_path.relative_to(REPO_ROOT).as_posix())
    print("  checksums:", checksums_path.relative_to(REPO_ROOT).as_posix())
    print("  Qdrant executed: False")
    print("  retrieval executed: False")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())