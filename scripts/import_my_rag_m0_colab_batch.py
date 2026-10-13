# ruff: noqa: E402
"""My_RAG M0-L3 import — Colab embeddings back to local."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tarfile
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from rag_system.embedding.models import ChunkEmbeddingRecord

OUT_ROOT = REPO / "outputs" / "external_my_rag"
COLAB_ROOT = OUT_ROOT / "_colab_v1"
DATASETS = ("clerk", "cuad", "greek_bar_bench", "lara")


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def write_json_atomic(p: Path, d: dict[str, Any]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(
        json.dumps(d, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tmp.replace(p)


def import_one(ds: str, out_bundle: Path, work: Path) -> dict[str, Any]:
    chunks_mf = json.loads(
        (OUT_ROOT / ds / "chunks-v1" / "manifest.json").read_text(encoding="utf-8")
    )
    expected_sha = chunks_mf["chunks_jsonl_sha256"]
    expected_count = chunks_mf["chunk_count"]

    extract = work / ds
    extract.mkdir(parents=True, exist_ok=True)
    with tarfile.open(out_bundle, "r:gz") as tar:
        tar.extractall(extract)

    recs_path = extract / "output" / "embedding_records.jsonl"
    man_path = extract / "output" / "embedding_manifest.json"
    if not recs_path.exists() or not man_path.exists():
        raise FileNotFoundError(f"Missing output files inside {out_bundle}")

    man = json.loads(man_path.read_text(encoding="utf-8"))
    if man["source_chunks_file_sha256"] != expected_sha:
        raise RuntimeError(f"{ds}: chunks SHA mismatch")

    records: list[ChunkEmbeddingRecord] = []
    seen: set[str] = set()
    with recs_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            # Strip debug fields not declared on ChunkEmbeddingRecord
            raw.pop("metadata", None)
            rec = ChunkEmbeddingRecord.model_validate(raw)
            if rec.chunk_id in seen:
                raise RuntimeError(f"{ds}: duplicate {rec.chunk_id}")
            seen.add(rec.chunk_id)
            records.append(rec)

    if len(records) != expected_count:
        raise RuntimeError(
            f"{ds}: got {len(records)} records, expected {expected_count}"
        )

    stage = OUT_ROOT / ds / "embeddings-v1"
    stage.mkdir(parents=True, exist_ok=True)
    recs_out = stage / "embedding_records.jsonl"
    with recs_out.open("w", encoding="utf-8", newline="\n") as fh:
        for rec in records:
            fh.write(rec.model_dump_json())
            fh.write("\n")

    recs_sha = sha256_file(recs_out)
    stage_manifest = {
        "artifact_type": "my_rag_m0_embedding_manifest",
        "artifact_version": "1.0",
        "status": "SUCCEEDED",
        "dataset_id": ds,
        "record_count": len(records),
        "dense_dimension": man["dense_dimension"],
        "dense_precision": man["dense_precision"],
        "dense_normalized": man["dense_normalized"],
        "embedding_kind": man["embedding_kind"],
        "model_id": man["model_id"],
        "model_revision": man["model_revision"],
        "chunker_config_hash": man["chunker_config_hash"],
        "embedding_config_hash": man["embedding_config_hash"],
        "source_chunks_file_sha256": man["source_chunks_file_sha256"],
        "embedding_records_sha256": recs_sha,
        "next_gate": "my_rag_m0_qdrant_ingest",
    }
    write_json_atomic(stage / "manifest.json", stage_manifest)
    write_json_atomic(
        stage / "checksums.json",
        {"files": {"embedding_records.jsonl": recs_sha}},
    )

    return {
        "dataset_id": ds,
        "record_count": len(records),
        "embedding_records_sha256": recs_sha,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="*", default=list(DATASETS))
    ap.add_argument("--out_dir", default=str(COLAB_ROOT))
    args = ap.parse_args()

    t0 = time.perf_counter()
    work = COLAB_ROOT / "_import_work"
    work.mkdir(parents=True, exist_ok=True)
    out_dir = Path(args.out_dir)

    results = []
    for ds in args.datasets:
        cand = out_dir / f"my_rag_colab_output_{ds}_v1.tar.gz"
        if not cand.exists():
            print(f"[skip] {cand.name} not found")
            continue
        print(f"[import] {ds} ...")
        try:
            results.append(import_one(ds, cand, work))
        except Exception as exc:
            print(f"  FAILED: {type(exc).__name__}: {exc}")
            results.append({"dataset_id": ds, "status": "FAILED",
                            "error": f"{type(exc).__name__}: {exc}"})

    write_json_atomic(COLAB_ROOT / "import_manifest.json", {
        "artifact_type": "my_rag_m0_import_manifest",
        "artifact_version": "1.0",
        "execution_id": "my-rag-m0-l3-import-v1",
        "status": "SUCCEEDED" if all(
            "record_count" in r for r in results
        ) else "PARTIAL",
        "datasets": results,
        "wall_clock_seconds": round(time.perf_counter() - t0, 3),
    })

    print()
    print("M0-L3 IMPORT SUMMARY")
    for r in results:
        if "record_count" in r:
            print(f"  {r['dataset_id']:18s} records={r['record_count']}")
        else:
            print(f"  {r['dataset_id']:18s} FAILED: {r.get('error')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())