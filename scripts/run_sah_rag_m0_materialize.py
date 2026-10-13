# ruff: noqa: E402
"""SAH-RAG M0-L2 - Parse + chunk canonical datasets for the 4-dataset scope."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

# Offline enforcement BEFORE imports
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from rag_system.chunking.hierarchical import HierarchicalChildChunker
from rag_system.chunking.huggingface import HuggingFaceOffsetTokenizer
from rag_system.chunking.pipeline import chunk_parsed_artifacts
from rag_system.parsing.deterministic import DeterministicStructuralParser
from rag_system.parsing.pipeline import parse_canonical_dataset

OUT_ROOT = REPO / "outputs" / "external_sah_rag"
SNAPSHOT = Path(
    "D:/rag_work/hf_cache/models--BAAI--bge-m3/snapshots/"
    "5617a9f61b028005a4858fdac845db406aefb181"
)
MODEL_ID = "BAAI/bge-m3"
REVISION = "5617a9f61b028005a4858fdac845db406aefb181"

DATASETS = ("clerk", "cuad", "greek_bar_bench", "lara")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def is_stage_done(stage_dir: Path) -> bool:
    mf = stage_dir / "manifest.json"
    if not mf.exists():
        return False
    try:
        d = json.loads(mf.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return d.get("status") == "SUCCEEDED"


def build_tokenizer() -> HuggingFaceOffsetTokenizer:
    from transformers import AutoTokenizer
    if not SNAPSHOT.exists():
        raise FileNotFoundError(f"Missing snapshot: {SNAPSHOT}")
    hf_tok = AutoTokenizer.from_pretrained(
        str(SNAPSHOT), local_files_only=True,
    )
    if not getattr(hf_tok, "is_fast", False):
        raise RuntimeError("HF tokenizer is not a fast tokenizer")
    return HuggingFaceOffsetTokenizer(
        hf_tok, model_id=MODEL_ID, revision=REVISION,
    )


def run_parse(
    dataset_id: str,
    parser: DeterministicStructuralParser,
    *,
    force: bool,
) -> dict[str, Any]:
    stage = OUT_ROOT / dataset_id / "parsed-v1"
    if not force and is_stage_done(stage):
        print(f"[{dataset_id}] parse already SUCCEEDED; skip")
        return json.loads((stage / "manifest.json").read_text(encoding="utf-8"))

    stage.mkdir(parents=True, exist_ok=True)
    parsed_path = stage / "parsed.jsonl"
    manifest_path = stage / "parsed.manifest.json"

    t0 = time.perf_counter()
    print(f"[{dataset_id}] parsing ...")
    summary = parse_canonical_dataset(
        dataset_id=dataset_id,
        parser=parser,
        output_path=parsed_path,
        manifest_path=manifest_path,
        overwrite=True,
        execution_profile="local_cpu_sah_rag_m0",
    )
    wall = time.perf_counter() - t0
    parsed_sha = sha256_file(parsed_path)

    manifest: dict[str, Any] = {
        "artifact_type": "sah_rag_m0_parse_manifest",
        "artifact_version": "1.0",
        "status": "SUCCEEDED",
        "dataset_id": dataset_id,
        "document_count": summary.document_count,
        "parent_count": summary.parent_count,
        "empty_document_count": summary.empty_document_count,
        "parsed_jsonl_sha256": parsed_sha,
        "wall_clock_seconds": round(wall, 3),
    }
    write_json_atomic(stage / "manifest.json", manifest)
    print(
        f"[{dataset_id}] parsed docs={summary.document_count} "
        f"parents={summary.parent_count} in {wall:.1f}s"
    )
    return manifest


def run_chunk(
    dataset_id: str,
    chunker: HierarchicalChildChunker,
    *,
    force: bool,
) -> dict[str, Any]:
    stage = OUT_ROOT / dataset_id / "chunks-v1"
    if not force and is_stage_done(stage):
        print(f"[{dataset_id}] chunk already SUCCEEDED; skip")
        return json.loads((stage / "manifest.json").read_text(encoding="utf-8"))

    stage.mkdir(parents=True, exist_ok=True)
    parsed_dir = OUT_ROOT / dataset_id / "parsed-v1"
    parsed_path = parsed_dir / "parsed.jsonl"
    parsed_manifest_path = parsed_dir / "parsed.manifest.json"
    chunks_path = stage / "chunks.jsonl"
    manifest_path = stage / "chunks.manifest.json"

    t0 = time.perf_counter()
    print(f"[{dataset_id}] chunking ...")
    summary = chunk_parsed_artifacts(
        input_path=parsed_path,
        output_path=chunks_path,
        chunker=chunker,
        manifest_path=manifest_path,
        input_manifest_path=parsed_manifest_path,
        dataset_id=dataset_id,
        overwrite=True,
        execution_profile="local_cpu_sah_rag_m0",
    )
    wall = time.perf_counter() - t0
    chunks_sha = sha256_file(chunks_path)

    manifest: dict[str, Any] = {
        "artifact_type": "sah_rag_m0_chunk_manifest",
        "artifact_version": "1.0",
        "status": "SUCCEEDED",
        "dataset_id": dataset_id,
        "parsed_document_count": summary.parsed_document_count,
        "parent_count": summary.parent_count,
        "chunk_count": summary.chunk_count,
        "parents_without_chunks": summary.parents_without_chunks,
        "chunks_jsonl_sha256": chunks_sha,
        "wall_clock_seconds": round(wall, 3),
        "chunker_config_hash": chunker.config_hash,
    }
    write_json_atomic(stage / "manifest.json", manifest)
    print(
        f"[{dataset_id}] chunks={summary.chunk_count} "
        f"parents={summary.parent_count} in {wall:.1f}s"
    )
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="*", default=list(DATASETS))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    print("[M0-L2] loading BGE-M3 tokenizer (offline) ...")
    offset_tok = build_tokenizer()
    print(f"[M0-L2] tokenizer_id = {offset_tok.tokenizer_id}")
    print(f"[M0-L2] special_tokens_count = {offset_tok.special_tokens_count}")

    chunker = HierarchicalChildChunker(tokenizer=offset_tok)
    print(f"[M0-L2] chunker version = {chunker.config.chunker_version}")
    print(f"[M0-L2] chunk_size = {chunker.config.chunk_size}")
    print(f"[M0-L2] chunk_overlap = {chunker.config.chunk_overlap}")
    print(f"[M0-L2] content_chunk_size = {chunker.content_chunk_size}")
    print(f"[M0-L2] minimum_final = {chunker.minimum_final_chunk_tokens}")
    print(f"[M0-L2] config_hash = {chunker.config_hash[:16]}")

    parser = DeterministicStructuralParser()
    print(f"[M0-L2] parser = {parser.config.parser_id} v{parser.config.parser_version}")
    print()

    results: list[dict[str, Any]] = []
    for ds in args.datasets:
        try:
            p = run_parse(ds, parser, force=args.force)
            c = run_chunk(ds, chunker, force=args.force)
            results.append({"dataset_id": ds, "parse": p, "chunk": c})
        except Exception as exc:
            print(f"[{ds}] FAILED: {type(exc).__name__}: {exc}")
            results.append({
                "dataset_id": ds,
                "status": "FAILED",
                "error": f"{type(exc).__name__}: {exc}",
            })

    cross = {
        "artifact_type": "sah_rag_m0_l2_cross_dataset",
        "artifact_version": "1.0",
        "execution_id": "sah-rag-m0-l2-v1",
        "status": "SUCCEEDED" if all(
            r.get("chunk", {}).get("status") == "SUCCEEDED"
            for r in results
        ) else "PARTIAL",
        "datasets": results,
    }
    write_json_atomic(OUT_ROOT / "_m0_l2_root_manifest.json", cross)

    print()
    print("M0-L2 SUMMARY")
    print("=" * 78)
    for r in results:
        c = r.get("chunk", {})
        if r.get("status") == "FAILED":
            print(f"  {r['dataset_id']:18s} FAILED: {r.get('error')}")
        else:
            print(
                f"  {r['dataset_id']:18s} "
                f"docs={c.get('parsed_document_count', '?'):>6} "
                f"parents={c.get('parent_count', '?'):>7} "
                f"chunks={c.get('chunk_count', '?'):>8} "
                f"secs={c.get('wall_clock_seconds')}"
            )

    total_chunks = sum(
        r.get("chunk", {}).get("chunk_count", 0) for r in results
    )
    total_parents = sum(
        r.get("chunk", {}).get("parent_count", 0) for r in results
    )
    print()
    print(f"TOTAL chunks={total_chunks} parents={total_parents}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())