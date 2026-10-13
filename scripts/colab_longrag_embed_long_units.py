"""LongRAG Colab-GPU embedding runner.

Runs on Google Colab. Reads one or more input bundles produced by
scripts/prepare_longrag_colab_embedding_batch.py, and writes output
bundles containing dense embedding records.

Dense-only BGE-M3, FP16 primary, CUDA. No network access beyond the
initial model snapshot materialization by the caller.

This file is shipped inside each input bundle and must not be imported
by the local repo.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tarfile
import time
from pathlib import Path
from typing import Any

BGE_M3_MODEL_ID = "BAAI/bge-m3"
BGE_M3_MODEL_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
BGE_M3_DENSE_DIMENSION = 1024
BGE_M3_SPECIAL_TOKEN_COUNT = 2
BGE_M3_MAX_INPUT_TOKENS = 8192

ADAPTER_ID = "longrag_retrieval_adapter"
ADAPTER_VERSION = "0.2.0-source-corrected"
SOURCE_ARCHITECTURE = "LongRAG"
SOURCE_ARCHITECTURE_COMMIT = "ba92f95"

DEFAULT_BATCH_SIZE = 16


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
        .encode("utf-8")
        + b"\n"
    )


def write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(canonical_json_bytes(data))
    tmp.replace(path)


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


def read_units(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_units_from_bundle(
    bundle_path: Path, work_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any], Path]:
    extract_dir = work_dir / bundle_path.stem
    extract_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(bundle_path, "r:gz") as tar:
        tar.extractall(extract_dir)

    units_path = extract_dir / "input" / "units.jsonl"
    manifest_path = extract_dir / "input" / "manifest.json"
    if not units_path.exists() or not manifest_path.exists():
        raise FileNotFoundError(
            f"Missing input files inside {bundle_path}"
        )

    units_sha = sha256_file(units_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_sha = manifest.get("units_jsonl_sha256")
    if expected_sha and units_sha != expected_sha:
        raise RuntimeError(
            f"SHA mismatch {bundle_path.name}: "
            f"{units_sha} != {expected_sha}"
        )

    units = read_units(units_path)
    return units, manifest, units_path


def embed_dataset(
    model: Any,
    units: list[dict[str, Any]],
    manifest: dict[str, Any],
    *,
    batch_size: int,
    max_length: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    dataset_id = manifest["dataset_id"]
    condition_id = manifest["condition_id"]
    target_token_size = manifest["target_token_size"]
    source_units_file_sha256 = manifest["units_jsonl_sha256"]
    embedding_config_hash = hashlib.sha256(
        f"{BGE_M3_MODEL_ID}:{BGE_M3_MODEL_REVISION}:dense:fp16:v1".encode()
    ).hexdigest()

    texts = [u["text"] for u in units]

    t0 = time.time()
    out = model.encode(
        texts,
        batch_size=batch_size,
        max_length=max_length,
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False,
    )
    dense = out["dense_vecs"]
    print(
        f"  encoded {len(texts)} units in {time.time()-t0:.1f}s "
        f"({len(texts)/(time.time()-t0):.1f} u/s)"
    )

    if len(dense) != len(units):
        raise RuntimeError(
            f"Embedding count mismatch: {len(dense)} != {len(units)}"
        )

    records: list[dict[str, Any]] = []
    for i, unit in enumerate(units):
        vec = [float(x) for x in dense[i]]
        if len(vec) != BGE_M3_DENSE_DIMENSION:
            raise RuntimeError(
                f"Dimension mismatch at index {i}: {len(vec)}"
            )
        rendered = int(unit["token_count"])
        actual = rendered + BGE_M3_SPECIAL_TOKEN_COUNT
        if actual > BGE_M3_MAX_INPUT_TOKENS:
            raise RuntimeError(
                f"Unit {unit['long_unit_id']}: actual tokens "
                f"{actual} > max {BGE_M3_MAX_INPUT_TOKENS}"
            )
        source_sha = hashlib.sha256(
            unit["text"].encode("utf-8")
        ).hexdigest()

        records.append(
            {
                "long_unit_id": unit["long_unit_id"],
                "condition_id": condition_id,
                "dataset_id": dataset_id,
                "construction_policy": unit["construction_policy"],
                "target_token_size": target_token_size,
                "maximum_model_input_tokens": BGE_M3_MAX_INPUT_TOKENS,
                "unit_order": int(unit["unit_order"]),
                "rendered_token_count": rendered,
                "special_token_count": BGE_M3_SPECIAL_TOKEN_COUNT,
                "actual_model_token_count": actual,
                "was_truncated": False,
                "member_doc_ids": list(unit["member_doc_ids"]),
                "member_parent_ids": list(unit["member_parent_ids"]),
                "member_chunk_ids": list(unit["member_chunk_ids"]),
                "model_id": BGE_M3_MODEL_ID,
                "model_revision": BGE_M3_MODEL_REVISION,
                "tokenizer_id": BGE_M3_MODEL_ID,
                "embedding_config_hash": embedding_config_hash,
                "dense_vector": vec,
                "dense_dimension": BGE_M3_DENSE_DIMENSION,
                "dense_precision": "float16",
                "dense_normalized": True,
                "source_unit_sha256": source_sha,
                "source_units_file_sha256": source_units_file_sha256,
                "adapter_id": ADAPTER_ID,
                "adapter_version": ADAPTER_VERSION,
                "source_architecture": SOURCE_ARCHITECTURE,
                "source_architecture_commit": SOURCE_ARCHITECTURE_COMMIT,
                "interface_validation": True,
                "paper_faithful_result": False,
                "metadata": {
                    "colab_dtype": "float16",
                    "colab_batch_size": batch_size,
                    "colab_max_length": max_length,
                },
            }
        )
    return records, {
        "embedding_config_hash": embedding_config_hash,
        "record_count": len(records),
        "dense_dimension": BGE_M3_DENSE_DIMENSION,
    }


def pack_output(
    dataset_id: str,
    records: list[dict[str, Any]],
    manifest_in: dict[str, Any],
    embed_meta: dict[str, Any],
    out_dir: Path,
) -> Path:
    stage = out_dir / f"stage_{dataset_id}"
    stage.mkdir(parents=True, exist_ok=True)
    records_path = stage / "embedding_records.jsonl"
    manifest_path = stage / "embedding_manifest.json"
    checksums_path = stage / "checksums.json"

    write_jsonl_atomic(records_path, records)

    records_sha = sha256_file(records_path)
    embed_manifest: dict[str, Any] = {
        "artifact_type": "longrag_colab_embedding_manifest",
        "artifact_version": "1.0",
        "dataset_id": dataset_id,
        "condition_id": manifest_in["condition_id"],
        "target_token_size": manifest_in["target_token_size"],
        "source_units_file_sha256": manifest_in["units_jsonl_sha256"],
        "model_id": BGE_M3_MODEL_ID,
        "model_revision": BGE_M3_MODEL_REVISION,
        "dense_dimension": BGE_M3_DENSE_DIMENSION,
        "dense_precision": "float16",
        "dense_normalized": True,
        "record_count": len(records),
        "embedding_records_sha256": records_sha,
        "embedding_config_hash": embed_meta["embedding_config_hash"],
        "execution_environment": "google_colab_gpu",
        "gpu_name": _gpu_name(),
        "wall_clock_hint": embed_meta.get("wall_clock_seconds"),
        "adapter_id": ADAPTER_ID,
        "adapter_version": ADAPTER_VERSION,
        "source_architecture": SOURCE_ARCHITECTURE,
        "source_architecture_commit": SOURCE_ARCHITECTURE_COMMIT,
    }
    write_json_atomic(manifest_path, embed_manifest)

    checksums = {
        "files": {
            "embedding_records.jsonl": records_sha,
            "embedding_manifest.json": sha256_file(manifest_path),
        }
    }
    write_json_atomic(checksums_path, checksums)

    out_bundle = out_dir / f"longrag_colab_output_{dataset_id}_v1.tar.gz"
    if out_bundle.exists():
        out_bundle.unlink()
    with tarfile.open(out_bundle, "w:gz") as tar:
        tar.add(records_path, arcname="output/embedding_records.jsonl")
        tar.add(manifest_path, arcname="output/embedding_manifest.json")
        tar.add(checksums_path, arcname="output/checksums.json")
    return out_bundle


def _gpu_name() -> str:
    try:
        import torch  # type: ignore[import-untyped]
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return "unknown"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "bundles", nargs="*", help="input bundle paths",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="process every longrag_colab_input_*.tar.gz in CWD",
    )
    parser.add_argument(
        "--out_dir", default="/content/longrag_colab_output",
    )
    parser.add_argument(
        "--work_dir", default="/content/longrag_work",
    )
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max_length", type=int, default=4098)
    args = parser.parse_args()

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    import torch  # type: ignore[import-untyped]
    if not torch.cuda.is_available():
        print("ERROR: CUDA is required.", file=sys.stderr)
        return 2
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    from FlagEmbedding import BGEM3FlagModel  # type: ignore[import-untyped]

    print("Loading BGE-M3 dense-only FP16 on CUDA ...")
    model = BGEM3FlagModel(
        BGE_M3_MODEL_ID,
        revision=BGE_M3_MODEL_REVISION,
        use_fp16=True,
        device="cuda",
    )

    if args.all:
        bundles = sorted(Path.cwd().glob("longrag_colab_input_*_v1.tar.gz"))
    else:
        bundles = [Path(b) for b in args.bundles]
    if not bundles:
        print("No input bundles found.", file=sys.stderr)
        return 2

    out_dir = Path(args.out_dir)
    work_dir = Path(args.work_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    summary: list[dict[str, Any]] = []
    for b in bundles:
        print(f"\n=== {b.name} ===")
        t0 = time.time()
        units, manifest_in, _ = load_units_from_bundle(b, work_dir)
        print(f"  loaded {len(units)} units")
        records, embed_meta = embed_dataset(
            model, units, manifest_in,
            batch_size=args.batch_size,
            max_length=args.max_length,
        )
        embed_meta["wall_clock_seconds"] = round(time.time() - t0, 2)
        out_bundle = pack_output(
            manifest_in["dataset_id"],
            records, manifest_in, embed_meta, out_dir,
        )
        summary.append(
            {
                "dataset_id": manifest_in["dataset_id"],
                "record_count": len(records),
                "out_bundle": str(out_bundle),
                "out_bundle_sha256": sha256_file(out_bundle),
                "out_bundle_bytes": out_bundle.stat().st_size,
                "seconds": embed_meta["wall_clock_seconds"],
            }
        )
        print(f"  -> {out_bundle.name} ({out_bundle.stat().st_size:,} bytes)")

    write_json_atomic(
        out_dir / "colab_summary.json",
        {
            "artifact_type": "longrag_colab_embedding_run_summary",
            "artifact_version": "1.0",
            "datasets": summary,
        },
    )
    print("\nCOLAB RUN COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())