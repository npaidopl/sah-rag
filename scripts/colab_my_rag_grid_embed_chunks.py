"""My_RAG M3 Colab runner — grid chunk embeddings.

Processes one bundle at a time (dataset, chunk_size, chunk_overlap).
Embeds with BGE-M3 dense + learned sparse, writes output bundle.
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

MODEL_ID = "BAAI/bge-m3"
MODEL_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
DENSE_DIMENSION = 1024
BATCH_SIZE = 32
MAX_LENGTH = 2048


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


def write_jsonl_atomic(p: Path, rows: list[dict[str, Any]]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False, sort_keys=True,
                                separators=(",", ":")))
            fh.write("\n")
    tmp.replace(p)


def read_jsonl(p: Path) -> list[dict[str, Any]]:
    rows = []
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def normalize_sparse(lex: dict[Any, Any]) -> tuple[list[int], list[float]]:
    items = [(int(k), float(v)) for k, v in lex.items() if float(v) != 0.0]
    items.sort(key=lambda p: p[0])
    return [i for i, _ in items], [v for _, v in items]


def extract_bundle(bundle: Path, work: Path) -> dict[str, Any]:
    extract_dir = work / bundle.stem
    extract_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(bundle, "r:gz") as tar:
        tar.extractall(extract_dir)

    meta_path = extract_dir / "input" / "grid_meta.json"
    chunks_path = extract_dir / "input" / "chunks.jsonl"
    if not meta_path.exists() or not chunks_path.exists():
        raise FileNotFoundError(f"Missing input files inside {bundle}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    actual_sha = sha256_file(chunks_path)
    if actual_sha != meta["chunks_jsonl_sha256"]:
        raise RuntimeError(f"SHA mismatch: {bundle.name}")

    return {
        "meta": meta,
        "chunks_path": chunks_path,
        "extract_dir": extract_dir,
    }


def embed_dataset(
    model: Any,
    chunks: list[dict[str, Any]],
    meta: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ds = meta["dataset_id"]
    cs = meta["chunk_size"]
    ov = meta["chunk_overlap"]
    chunker_hash = meta["chunker_config_hash"]

    config_hash = hashlib.sha256(
        f"{MODEL_ID}:{MODEL_REVISION}:hybrid:fp16:gpu:"
        f"c{cs}_o{ov}:{chunker_hash}".encode()
    ).hexdigest()

    texts = [c["text_normalized"] for c in chunks]

    t0 = time.time()
    out = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        max_length=MAX_LENGTH,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False,
    )
    dense = out["dense_vecs"]
    sparse = out["lexical_weights"]
    elapsed = time.time() - t0
    rate = len(texts) / max(elapsed, 1e-9)
    print(
        f"  encoded {len(texts)} chunks in {elapsed:.1f}s "
        f"({rate:.1f} ch/s)"
    )

    if len(dense) != len(chunks) or len(sparse) != len(chunks):
        raise RuntimeError(f"count mismatch: {len(dense)} {len(sparse)}")

    records = []
    for i, ch in enumerate(chunks):
        dv = [float(x) for x in dense[i]]
        if len(dv) != DENSE_DIMENSION:
            raise RuntimeError(f"dim {len(dv)} at index {i}")
        sidx, sval = normalize_sparse(sparse[i])
        token_count = int(ch.get("token_count", 0))
        if token_count <= 0:
            raise RuntimeError(f"bad token_count at {ch['chunk_id']}")
        records.append({
            "chunk_id": ch["chunk_id"],
            "parent_id": ch["parent_id"],
            "doc_id": ch["doc_id"],
            "dataset": ds,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "tokenizer_id": MODEL_ID,
            "embedding_config_hash": config_hash,
            "dense_vector": dv,
            "sparse_vector": {"indices": sidx, "values": sval},
            "dense_dimension": DENSE_DIMENSION,
            "dense_precision": "float16",
            "dense_normalized": True,
        })

    return records, {
        "embedding_config_hash": config_hash,
        "record_count": len(records),
        "wall_clock_seconds": round(elapsed, 3),
    }


def pack_output(
    meta: dict[str, Any],
    records: list[dict[str, Any]],
    embed_meta: dict[str, Any],
    out_dir: Path,
) -> Path:
    ds = meta["dataset_id"]
    cs = meta["chunk_size"]
    ov = meta["chunk_overlap"]

    stage = out_dir / f"stage_{ds}_c{cs}_o{ov}"
    stage.mkdir(parents=True, exist_ok=True)
    recs_path = stage / "embedding_records.jsonl"
    man_path = stage / "embedding_manifest.json"
    cks_path = stage / "checksums.json"

    write_jsonl_atomic(recs_path, records)
    recs_sha = sha256_file(recs_path)

    embed_manifest = {
        "artifact_type": "my_rag_grid_embedding_manifest",
        "artifact_version": "1.0",
        "dataset_id": ds,
        "chunk_size": cs,
        "chunk_overlap": ov,
        "source_chunks_file_sha256": meta["chunks_jsonl_sha256"],
        "chunker_config_hash": meta["chunker_config_hash"],
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "dense_dimension": DENSE_DIMENSION,
        "dense_precision": "float16",
        "dense_normalized": True,
        "embedding_kind": "hybrid",
        "record_count": len(records),
        "embedding_records_sha256": recs_sha,
        "embedding_config_hash": embed_meta["embedding_config_hash"],
        "execution_environment": "google_colab_gpu",
        "wall_clock_seconds": embed_meta["wall_clock_seconds"],
    }
    write_json_atomic(man_path, embed_manifest)
    write_json_atomic(cks_path, {"files": {
        "embedding_records.jsonl": recs_sha,
        "embedding_manifest.json": sha256_file(man_path),
    }})

    out_bundle = out_dir / (
        f"my_rag_grid_output_{ds}_c{cs}_o{ov}_v1.tar.gz"
    )
    if out_bundle.exists():
        out_bundle.unlink()
    with tarfile.open(out_bundle, "w:gz") as tar:
        tar.add(recs_path, arcname="output/embedding_records.jsonl")
        tar.add(man_path, arcname="output/embedding_manifest.json")
        tar.add(cks_path, arcname="output/checksums.json")
    return out_bundle


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("bundles", nargs="*")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--out_dir", default="/content/grid_out")
    ap.add_argument("--work_dir", default="/content/grid_work")
    args = ap.parse_args()

    os.environ.setdefault("HF_HUB_OFFLINE", "0")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "0")

    import torch
    if not torch.cuda.is_available():
        print("CUDA required", file=sys.stderr)
        return 2
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    from FlagEmbedding import BGEM3FlagModel

    print("Loading BGE-M3 hybrid FP16 ...")
    model = BGEM3FlagModel(
        MODEL_ID, revision=MODEL_REVISION, use_fp16=True, device="cuda",
    )

    if args.all:
        bundles = sorted(
            Path.cwd().glob("my_rag_grid_input_*_v1.tar.gz")
        )
    else:
        bundles = [Path(b) for b in args.bundles]
    if not bundles:
        print("No bundles.", file=sys.stderr)
        return 2

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    for b in bundles:
        print(f"\n=== {b.name} ===")
        t0 = time.time()
        info = extract_bundle(b, work_dir)
        chunks = read_jsonl(info["chunks_path"])
        meta = info["meta"]
        print(
            f"  {meta['dataset_id']} c{meta['chunk_size']}_"
            f"o{meta['chunk_overlap']} chunks={len(chunks)}"
        )
        records, embed_meta = embed_dataset(model, chunks, meta)
        out_bundle = pack_output(meta, records, embed_meta, out_dir)
        summary.append({
            "dataset_id": meta["dataset_id"],
            "chunk_size": meta["chunk_size"],
            "chunk_overlap": meta["chunk_overlap"],
            "record_count": len(records),
            "out_bundle": str(out_bundle),
            "out_bundle_sha256": sha256_file(out_bundle),
            "out_bundle_bytes": out_bundle.stat().st_size,
            "seconds": round(time.time() - t0, 1),
        })
        print(
            f"  -> {out_bundle.name} "
            f"({out_bundle.stat().st_size/1e6:.1f} MB)"
        )

    write_json_atomic(out_dir / "colab_summary.json", {
        "artifact_type": "my_rag_grid_colab_run_summary",
        "artifact_version": "1.0",
        "datasets": summary,
    })
    print("\nCOLAB GRID RUN COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())