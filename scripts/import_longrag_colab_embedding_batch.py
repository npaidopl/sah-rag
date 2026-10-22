# ruff: noqa: E402
"""LongRAG L3 - Import Colab embedding output bundles.

Validates each output bundle against the local input units bundle and
stores the accepted records under outputs/external_longrag/<ds>/embeddings-v1/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tarfile
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from baselines.longrag_embeddings import (  # noqa: E402
    LongUnitEmbeddingRecord,
)

OUTPUT_ROOT = REPO_ROOT / "outputs" / "external_longrag"
COLAB_ROOT = OUTPUT_ROOT / "_colab_v1"

EXPECTED_DATASETS: tuple[str, ...] = (
    "ruler",
    "loong",
    "long_bench_v2",
    "lara",
    "greek_bar_bench",
    "cuad",
    "clerk",
)


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


def atomic_write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(content)
    tmp.replace(path)


def write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    atomic_write_bytes(path, canonical_json_bytes(data))


def import_one(dataset_id: str, out_bundle: Path, work_dir: Path) -> dict[str, Any]:
    units_manifest = json.loads(
        (OUTPUT_ROOT / dataset_id / "units-v1" / "manifest.json")
        .read_text(encoding="utf-8")
    )
    expected_units_sha = units_manifest["units_jsonl_sha256"]
    expected_count = units_manifest["long_unit_count"]

    extract = work_dir / dataset_id
    extract.mkdir(parents=True, exist_ok=True)
    with tarfile.open(out_bundle, "r:gz") as tar:
        tar.extractall(extract)

    recs_path = extract / "output" / "embedding_records.jsonl"
    man_path = extract / "output" / "embedding_manifest.json"
    if not recs_path.exists() or not man_path.exists():
        raise FileNotFoundError(
            f"Missing output files inside {out_bundle}"
        )

    man = json.loads(man_path.read_text(encoding="utf-8"))
    if man["source_units_file_sha256"] != expected_units_sha:
        raise RuntimeError(
            f"{dataset_id}: unit file SHA mismatch "
            f"({man['source_units_file_sha256']} != {expected_units_sha})"
        )

    records: list[dict[str, Any]] = []
    with recs_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if len(records) != expected_count:
        raise RuntimeError(
            f"{dataset_id}: record count {len(records)} != expected {expected_count}"
        )

    # full schema validation via Pydantic
    seen: set[str] = set()
    validated: list[LongUnitEmbeddingRecord] = []
    for r in records:
        rec = LongUnitEmbeddingRecord.model_validate(r)
        if rec.long_unit_id in seen:
            raise RuntimeError(f"{dataset_id}: duplicate {rec.long_unit_id}")
        seen.add(rec.long_unit_id)
        validated.append(rec)

    stage = OUTPUT_ROOT / dataset_id / "embeddings-v1"
    stage.mkdir(parents=True, exist_ok=True)
    recs_out = stage / "embedding_records.jsonl"
    with recs_out.open("w", encoding="utf-8", newline="\n") as fh:
        for rec in validated:
            fh.write(rec.model_dump_json())
            fh.write("\n")

    recs_sha = sha256_file(recs_out)
    stage_manifest: dict[str, Any] = {
        "artifact_type": "longrag_full_track_embedding_stage_manifest",
        "artifact_version": "1.0",
        "execution_id": f"longrag-l3-import-{dataset_id}-v1",
        "status": "SUCCEEDED",
        "dataset_id": dataset_id,
        "record_count": len(validated),
        "dense_dimension": man["dense_dimension"],
        "dense_precision": man["dense_precision"],
        "model_id": man["model_id"],
        "model_revision": man["model_revision"],
        "embedding_config_hash": man["embedding_config_hash"],
        "source_units_file_sha256": man["source_units_file_sha256"],
        "embedding_records_sha256": recs_sha,
        "gpu_name": man.get("gpu_name"),
        "next_gate": "longrag_full_track_qdrant_ingest",
    }
    write_json_atomic(stage / "manifest.json", stage_manifest)
    write_json_atomic(
        stage / "checksums.json",
        {"files": {"embedding_records.jsonl": recs_sha}},
    )

    return {
        "dataset_id": dataset_id,
        "record_count": len(validated),
        "embedding_records_sha256": recs_sha,
        "out_bundle_sha256": sha256_file(out_bundle),
    }


def run(out_dir: Path, datasets: tuple[str, ...]) -> int:
    t0 = time.perf_counter()
    work = COLAB_ROOT / "_import_work"
    work.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    for ds in datasets:
        candidate = out_dir / f"longrag_colab_output_{ds}_v1.tar.gz"
        if not candidate.exists():
            print(f"[skip] {candidate.name} not found")
            continue
        print(f"[import] {ds} ...")
        try:
            results.append(import_one(ds, candidate, work))
        except Exception as exc:
            print(f"  FAILED {ds}: {exc}")
            results.append(
                {"dataset_id": ds, "status": "FAILED", "error": str(exc)}
            )

    write_json_atomic(
        COLAB_ROOT / "import_manifest.json",
        {
            "artifact_type": "longrag_colab_embedding_import_manifest",
            "artifact_version": "1.0",
            "execution_id": "longrag-l3-import-v1",
            "status": "SUCCEEDED"
            if all(
                r.get("status", "SUCCEEDED") == "SUCCEEDED" for r in results
            )
            else "PARTIAL",
            "datasets": results,
            "wall_clock_seconds": round(time.perf_counter() - t0, 3),
        },
    )
    print()
    print("L3 IMPORT SUMMARY")
    for r in results:
        print(
            f"  {r['dataset_id']:18s} records={r.get('record_count')} "
            f"status={r.get('status', 'SUCCEEDED')}"
        )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out_dir", default=str(COLAB_ROOT),
    )
    parser.add_argument(
        "--datasets", nargs="*", default=list(EXPECTED_DATASETS),
        choices=list(EXPECTED_DATASETS),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return run(Path(args.out_dir), tuple(args.datasets))


if __name__ == "__main__":
    raise SystemExit(main())