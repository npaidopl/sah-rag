# ruff: noqa: E402
"""SAH-RAG M2 - Materialize the 9-variant chunking grid.

For each (dataset, chunk_size, chunk_overlap) in the frozen grid:
  1. reuse the parsed-v1/parsed.jsonl from M0-L2 (parser is invariant)
  2. build a HierarchicalChildChunker with the specific config
  3. chunk the parsed documents
  4. write chunks.jsonl + grid manifest + checksums

Resume-safe: a variant marked SUCCEEDED in its manifest is skipped.
Byte-identity check: the (256, 32) variant must match M0-L2's
chunks-v1/chunks.jsonl byte-for-byte.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from rag_system.chunking.config import ChildChunkerConfig
from rag_system.chunking.hierarchical import HierarchicalChildChunker
from rag_system.chunking.huggingface import HuggingFaceOffsetTokenizer
from rag_system.chunking.pipeline import chunk_parsed_artifacts

OUT_ROOT = REPO / "outputs" / "external_sah_rag"
SNAPSHOT = Path(
    "D:/rag_work/hf_cache/models--BAAI--bge-m3/snapshots/"
    "5617a9f61b028005a4858fdac845db406aefb181"
)
MODEL_ID = "BAAI/bge-m3"
REVISION = "5617a9f61b028005a4858fdac845db406aefb181"

DATASETS = ("clerk", "cuad", "greek_bar_bench", "lara")
GRID: tuple[tuple[int, int], ...] = (
    (256, 0),
    (256, 32),
    (256, 128),
    (512, 0),
    (512, 32),
    (512, 128),
    (1024, 0),
    (1024, 32),
    (1024, 128),
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


def build_tokenizer() -> HuggingFaceOffsetTokenizer:
    from transformers import AutoTokenizer
    if not SNAPSHOT.exists():
        raise FileNotFoundError(f"Missing snapshot: {SNAPSHOT}")
    hf_tok = AutoTokenizer.from_pretrained(
        str(SNAPSHOT), local_files_only=True,
    )
    return HuggingFaceOffsetTokenizer(
        hf_tok, model_id=MODEL_ID, revision=REVISION,
    )


def variant_dir(ds: str, cs: int, ov: int) -> Path:
    return OUT_ROOT / ds / "grid-chunks-v1" / f"c{cs}_o{ov}"


def is_done(stage: Path) -> bool:
    mf = stage / "manifest.json"
    if not mf.exists():
        return False
    try:
        d = json.loads(mf.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return d.get("status") == "SUCCEEDED"


def run_variant(
    ds: str,
    cs: int,
    ov: int,
    tokenizer: HuggingFaceOffsetTokenizer,
    *,
    force: bool,
) -> dict[str, Any]:
    stage = variant_dir(ds, cs, ov)
    if not force and is_done(stage):
        print(f"[{ds}] c{cs}_o{ov} already SUCCEEDED; skip")
        return json.loads((stage / "manifest.json").read_text(encoding="utf-8"))

    parsed_dir = OUT_ROOT / ds / "parsed-v1"
    parsed_path = parsed_dir / "parsed.jsonl"
    parsed_mf = parsed_dir / "parsed.manifest.json"
    if not parsed_path.exists() or not parsed_mf.exists():
        raise FileNotFoundError(
            f"Missing parsed-v1 for {ds}: run M0-L2 first"
        )

    stage.mkdir(parents=True, exist_ok=True)
    config = ChildChunkerConfig(chunk_size=cs, chunk_overlap=ov)
    chunker = HierarchicalChildChunker(tokenizer=tokenizer, config=config)

    chunks_path = stage / "chunks.jsonl"
    manifest_pipeline_path = stage / "chunks.manifest.json"

    t0 = time.perf_counter()
    print(
        f"[{ds}] chunking c{cs}_o{ov} "
        f"(config_hash={chunker.config_hash[:16]}) ..."
    )
    summary = chunk_parsed_artifacts(
        input_path=parsed_path,
        output_path=chunks_path,
        chunker=chunker,
        manifest_path=manifest_pipeline_path,
        input_manifest_path=parsed_mf,
        dataset_id=ds,
        overwrite=True,
        execution_profile="local_cpu_sah_rag_m2",
    )
    wall = time.perf_counter() - t0
    chunks_sha = sha256_file(chunks_path)

    manifest: dict[str, Any] = {
        "artifact_type": "sah_rag_m2_grid_chunk_manifest",
        "artifact_version": "1.0",
        "status": "SUCCEEDED",
        "dataset_id": ds,
        "chunk_size": cs,
        "chunk_overlap": ov,
        "chunker_config_hash": chunker.config_hash,
        "content_chunk_size": chunker.content_chunk_size,
        "minimum_final_chunk_tokens": chunker.minimum_final_chunk_tokens,
        "parsed_document_count": summary.parsed_document_count,
        "parent_count": summary.parent_count,
        "chunk_count": summary.chunk_count,
        "parents_without_chunks": summary.parents_without_chunks,
        "chunks_jsonl_sha256": chunks_sha,
        "wall_clock_seconds": round(wall, 3),
    }
    write_json_atomic(stage / "manifest.json", manifest)
    write_json_atomic(
        stage / "checksums.json",
        {"files": {"chunks.jsonl": chunks_sha}},
    )
    print(
        f"[{ds}] c{cs}_o{ov} DONE chunks={summary.chunk_count} "
        f"in {wall:.1f}s"
    )
    return manifest


def verify_baseline_identical() -> None:
    """(256, 32) must match M0-L2's chunks-v1 byte-for-byte."""
    print("\n[verify] baseline (256,32) byte-identity check ...")
    for ds in DATASETS:
        baseline = OUT_ROOT / ds / "chunks-v1" / "chunks.jsonl"
        grid = variant_dir(ds, 256, 32) / "chunks.jsonl"
        if not baseline.exists() or not grid.exists():
            print(f"  {ds:18s} SKIP (missing)")
            continue
        a = sha256_file(baseline)
        b = sha256_file(grid)
        match = a == b
        print(
            f"  {ds:18s} baseline={a[:16]} grid={b[:16]} "
            f"match={match}"
        )
        if not match:
            raise RuntimeError(
                f"{ds}: baseline (256,32) not byte-identical to M0-L2"
            )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="*", default=list(DATASETS))
    ap.add_argument(
        "--variants", nargs="*", default=None,
        help="specific variants as 'cs_ov' (e.g. 256_32)",
    )
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    print("[M2] loading BGE-M3 tokenizer (offline) ...")
    tokenizer = build_tokenizer()
    print(f"[M2] tokenizer_id = {tokenizer.tokenizer_id}")
    print()

    if args.variants:
        grid: tuple[tuple[int, int], ...] = tuple(
            (int(v.split("_")[0]), int(v.split("_")[1]))
            for v in args.variants
        )
    else:
        grid = GRID

    results: list[dict[str, Any]] = []
    for ds in args.datasets:
        for cs, ov in grid:
            try:
                m = run_variant(
                    ds, cs, ov, tokenizer, force=args.force,
                )
                results.append({
                    "dataset_id": ds,
                    "chunk_size": cs,
                    "chunk_overlap": ov,
                    "chunk_count": m["chunk_count"],
                    "wall_clock_seconds": m["wall_clock_seconds"],
                    "status": "SUCCEEDED",
                })
            except Exception as exc:
                print(
                    f"[{ds}] c{cs}_o{ov} FAILED: "
                    f"{type(exc).__name__}: {exc}"
                )
                results.append({
                    "dataset_id": ds,
                    "chunk_size": cs,
                    "chunk_overlap": ov,
                    "status": "FAILED",
                    "error": str(exc),
                })

    # baseline identity check
    verify_baseline_identical()

    # write root manifest
    root = {
        "artifact_type": "sah_rag_m2_grid_root_manifest",
        "artifact_version": "1.0",
        "execution_id": "sah-rag-m2-v1",
        "status": "SUCCEEDED"
        if all(r["status"] == "SUCCEEDED" for r in results)
        else "PARTIAL",
        "variants": results,
        "grid_size": len(grid),
        "dataset_count": len(args.datasets),
    }
    write_json_atomic(OUT_ROOT / "_m2_grid_root_manifest.json", root)

    print()
    print("M2 SUMMARY")
    print("=" * 92)
    print(
        f"{'dataset':18s} {'variant':10s} {'chunks':>10s} "
        f"{'secs':>8s} {'status':>10s}"
    )
    for r in results:
        print(
            f"{r['dataset_id']:18s} "
            f"c{r['chunk_size']}_o{r['chunk_overlap']:<6d} "
            f"{r.get('chunk_count', 0):>10d} "
            f"{r.get('wall_clock_seconds', 0):>8.1f} "
            f"{r['status']:>10s}"
        )
    return 0 if root["status"] == "SUCCEEDED" else 1


if __name__ == "__main__":
    raise SystemExit(main())