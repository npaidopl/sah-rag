"""My_RAG M0 Colab-GPU embedding runner (dense + sparse, FP16).

Reads one or more my_rag_colab_input_*.tar.gz bundles, embeds all
child chunks with BGE-M3 (dense + learned-sparse), writes output
bundles. Chunks are short (<= 256 tokens) so this is fast on T4.
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
SPECIAL_TOKENS = 2
MAX_INPUT_TOKENS = 258  # 254 content + 2 special (chunk_size=256)

BATCH_SIZE = 32
MAX_LENGTH = 512


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


def extract_bundle(bundle: Path, work: Path) -> tuple[Path, Path, dict[str, Any]]:
    extract_dir = work / bundle.stem
    extract_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(bundle, "r:gz") as tar:
        tar.extractall(extract_dir)
    chunks = extract_dir / "input" / "chunks.jsonl"
    manifest_in = extract_dir / "input" / "chunks_manifest.json"
    if not chunks.exists() or not manifest_in.exists():
        raise FileNotFoundError(f"Missing input files inside {bundle}")
    manifest = json.loads(manifest_in.read_text(encoding="utf-8"))
    expected_sha = manifest.get("chunks_jsonl_sha256")
    if expected_sha and sha256_file(chunks) != expected_sha:
        raise RuntimeError(f"SHA mismatch for {bundle.name}")
    return chunks, manifest_in, manifest


def normalize_sparse(lex: dict[Any, Any]) -> tuple[list[int], list[float]]:
    items = [
        (int(k), float(v)) for k, v in lex.items() if float(v) != 0.0
    ]
    items.sort(key=lambda p: p[0])
    return [i for i, _ in items], [v for _, v in items]


def embed_dataset(
    model: Any, chunks: list[dict[str, Any]], manifest_in: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ds = manifest_in["dataset_id"]
    chunker_hash = manifest_in["chunker_config_hash"]
    config_hash = hashlib.sha256(
        f"{MODEL_ID}:{MODEL_REVISION}:hybrid:fp16:gpu:{chunker_hash}".encode()
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
    print(f"  encoded {len(texts)} chunks in {elapsed:.1f}s ({len(texts)/elapsed:.1f} ch/s)")

    if len(dense) != len(chunks) or len(sparse) != len(chunks):
        raise RuntimeError(
            f"count mismatch dense={len(dense)} sparse={len(sparse)} chunks={len(chunks)}"
        )

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
            "metadata": {
                "colab_dtype": "float16",
                "colab_batch_size": BATCH_SIZE,
                "colab_max_length": MAX_LENGTH,
                "source_chunk_token_count": token_count,
            },
        })
    meta = {
        "embedding_config_hash": config_hash,
        "record_count": len(records),
        "dense_dimension": DENSE_DIMENSION,
        "wall_clock_seconds": round(elapsed, 3),
    }
    return records, meta


def pack_output(ds: str, records: list[dict[str, Any]], manifest_in: dict[str, Any],
                meta: dict[str, Any], out_dir: Path) -> Path:
    stage = out_dir / f"stage_{ds}"
    stage.mkdir(parents=True, exist_ok=True)
    recs_path = stage / "embedding_records.jsonl"
    man_path = stage / "embedding_manifest.json"
    cks_path = stage / "checksums.json"

    write_jsonl_atomic(recs_path, records)
    recs_sha = sha256_file(recs_path)

    embed_manifest = {
        "artifact_type": "my_rag_m0_colab_embedding_manifest",
        "artifact_version": "1.0",
        "dataset_id": ds,
        "source_chunks_file_sha256": manifest_in["chunks_jsonl_sha256"],
        "chunker_config_hash": manifest_in["chunker_config_hash"],
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "dense_dimension": DENSE_DIMENSION,
        "dense_precision": "float16",
        "dense_normalized": True,
        "embedding_kind": "hybrid",
        "record_count": len(records),
        "embedding_records_sha256": recs_sha,
        "embedding_config_hash": meta["embedding_config_hash"],
        "execution_environment": "google_colab_gpu",
        "wall_clock_seconds": meta["wall_clock_seconds"],
    }
    write_json_atomic(man_path, embed_manifest)
    write_json_atomic(cks_path, {"files": {
        "embedding_records.jsonl": recs_sha,
        "embedding_manifest.json": sha256_file(man_path),
    }})

    out_bundle = out_dir / f"my_rag_colab_output_{ds}_v1.tar.gz"
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
    ap.add_argument("--out_dir", default="/content/myrag_output")
    ap.add_argument("--work_dir", default="/content/myrag_work")
    args = ap.parse_args()

    os.environ.setdefault("HF_HUB_OFFLINE", "0")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "0")

    import torch
    if not torch.cuda.is_available():
        print("ERROR: CUDA required.", file=sys.stderr)
        return 2
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    from FlagEmbedding import BGEM3FlagModel

    print("Loading BGE-M3 (hybrid, FP16, CUDA) ...")
    model = BGEM3FlagModel(
        MODEL_ID, revision=MODEL_REVISION, use_fp16=True, device="cuda",
    )

    if args.all:
        bundles = sorted(Path.cwd().glob("my_rag_colab_input_*_v1.tar.gz"))
    else:
        bundles = [Path(b) for b in args.bundles]
    if not bundles:
        print("No bundles found.", file=sys.stderr)
        return 2

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    for b in bundles:
        print(f"\n=== {b.name} ===")
        t0 = time.time()
        chunks_path, _, manifest_in = extract_bundle(b, work_dir)
        chunks = read_jsonl(chunks_path)
        print(f"  loaded {len(chunks)} chunks")
        records, meta = embed_dataset(model, chunks, manifest_in)
        out_bundle = pack_output(manifest_in["dataset_id"], records, manifest_in, meta, out_dir)
        summary.append({
            "dataset_id": manifest_in["dataset_id"],
            "record_count": len(records),
            "out_bundle": str(out_bundle),
            "out_bundle_sha256": sha256_file(out_bundle),
            "out_bundle_bytes": out_bundle.stat().st_size,
            "seconds": round(time.time() - t0, 1),
        })
        print(f"  -> {out_bundle.name} ({out_bundle.stat().st_size/1e6:.1f} MB)")

    write_json_atomic(out_dir / "colab_summary.json", {
        "artifact_type": "my_rag_m0_colab_run_summary",
        "artifact_version": "1.0",
        "datasets": summary,
    })
    print("\nCOLAB RUN COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())