# ruff: noqa: E402
"""My_RAG M3 import — grid embeddings back to local."""
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
COLAB_ROOT = OUT_ROOT / "_grid_colab_v1"
DATASETS = ("clerk", "cuad", "greek_bar_bench", "lara")
GRID: tuple[tuple[int, int], ...] = (
    (256, 0), (256, 32), (256, 128),
    (512, 0), (512, 32), (512, 128),
    (1024, 0), (1024, 32), (1024, 128),
)


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


def import_one(
    ds: str, cs: int, ov: int, bundle: Path, work: Path,
) -> dict[str, Any]:
    chunks_mf_path = (
        OUT_ROOT / ds / "grid-chunks-v1" / f"c{cs}_o{ov}" / "manifest.json"
    )
    chunks_mf = json.loads(chunks_mf_path.read_text(encoding="utf-8"))
    expected_sha = chunks_mf["chunks_jsonl_sha256"]
    expected_count = chunks_mf["chunk_count"]

    extract = work / f"{ds}_c{cs}_o{ov}"
    extract.mkdir(parents=True, exist_ok=True)
    with tarfile.open(bundle, "r:gz") as tar:
        tar.extractall(extract)

    recs_path = extract / "output" / "embedding_records.jsonl"
    man_path = extract / "output" / "embedding_manifest.json"
    if not recs_path.exists() or not man_path.exists():
        raise FileNotFoundError(f"Missing output files inside {bundle}")

    man = json.loads(man_path.read_text(encoding="utf-8"))
    if man["source_chunks_file_sha256"] != expected_sha:
        raise RuntimeError(f"{ds} c{cs}_o{ov}: chunks SHA mismatch")

    records: list[ChunkEmbeddingRecord] = []
    seen: set[str] = set()
    with recs_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            raw.pop("metadata", None)
            rec = ChunkEmbeddingRecord.model_validate(raw)
            if rec.chunk_id in seen:
                raise RuntimeError(
                    f"{ds} c{cs}_o{ov}: duplicate {rec.chunk_id}"
                )
            seen.add(rec.chunk_id)
            records.append(rec)

    if len(records) != expected_count:
        raise RuntimeError(
            f"{ds} c{cs}_o{ov}: got {len(records)} "
            f"expected {expected_count}"
        )

    stage = (
        OUT_ROOT / ds / "grid-embeddings-v1" / f"c{cs}_o{ov}"
    )
    stage.mkdir(parents=True, exist_ok=True)
    recs_out = stage / "embedding_records.jsonl"
    with recs_out.open("w", encoding="utf-8", newline="\n") as fh:
        for rec in records:
            fh.write(rec.model_dump_json())
            fh.write("\n")

    recs_sha = sha256_file(recs_out)
    stage_manifest = {
        "artifact_type": "my_rag_grid_embedding_manifest",
        "artifact_version": "1.0",
        "status": "SUCCEEDED",
        "dataset_id": ds,
        "chunk_size": cs,
        "chunk_overlap": ov,
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
    }
    write_json_atomic(stage / "manifest.json", stage_manifest)
    write_json_atomic(
        stage / "checksums.json",
        {"files": {"embedding_records.jsonl": recs_sha}},
    )

    return {
        "dataset_id": ds,
        "chunk_size": cs,
        "chunk_overlap": ov,
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

    results: list[dict[str, Any]] = []
    for ds in args.datasets:
        for cs, ov in GRID:
            cand = out_dir / (
                f"my_rag_grid_output_{ds}_c{cs}_o{ov}_v1.tar.gz"
            )
            if not cand.exists():
                print(f"[skip] {cand.name}")
                continue
            print(f"[import] {ds} c{cs}_o{ov} ...")
            try:
                results.append(import_one(ds, cs, ov, cand, work))
            except Exception as exc:
                print(f"  FAILED: {type(exc).__name__}: {exc}")
                results.append({
                    "dataset_id": ds,
                    "chunk_size": cs,
                    "chunk_overlap": ov,
                    "status": "FAILED",
                    "error": f"{type(exc).__name__}: {exc}",
                })

    write_json_atomic(COLAB_ROOT / "import_manifest.json", {
        "artifact_type": "my_rag_grid_import_manifest",
        "artifact_version": "1.0",
        "execution_id": "my-rag-m3-import-v1",
        "status": "SUCCEEDED",
        "record_count": len(results),
        "datasets": results,
        "wall_clock_seconds": round(time.perf_counter() - t0, 3),
    })

    print()
    print("M3 IMPORT SUMMARY")
    for r in results:
        if "record_count" in r:
            print(
                f"  {r['dataset_id']:18s} c{r['chunk_size']}_"
                f"o{r['chunk_overlap']:<4d} records={r['record_count']}"
            )
        else:
            print(
                f"  {r['dataset_id']:18s} c{r['chunk_size']}_"
                f"o{r['chunk_overlap']:<4d} FAILED"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())