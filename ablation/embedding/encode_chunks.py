"""Encode all chunks for one (variant, dataset). CLI entry.

Usage:
  python -m ablation.embedding.encode_chunks --variant A4_full_sah --dataset clerk
  python -m ablation.embedding.encode_chunks --tier 1
"""
import argparse
import json
import time
from pathlib import Path

from ablation.config import CHUNKS_DIR
from ablation.embedding.bge_m3 import BGEM3Embedder
from ablation.embedding.cache import EmbeddingCache
from ablation.orchestration.planner import build_plan
from ablation.retrieval.variant_specs import VARIANT_SPECS


def encode_one(variant: str, dataset: str, embedder: BGEM3Embedder,
               batch_size: int = 32, force: bool = False) -> dict:
    spec = VARIANT_SPECS.get(variant)
    if spec is None:
        return {"status": "skipped", "reason": f"unknown variant {variant}"}
    # Only variants with their OWN embed_source need encoding.
    # Tier 2 variants reuse A4, so no need to re-encode.
    if spec["embed_source"] != variant:
        return {"status": "skipped", "reason": f"reuses {spec['embed_source']}"}

    chunks_path = CHUNKS_DIR / variant / dataset / "chunks.jsonl"
    if not chunks_path.is_file():
        return {"status": "skipped", "reason": f"no chunks at {chunks_path}"}

    cache = EmbeddingCache(variant=variant, dataset=dataset)
    if cache.exists() and not force:
        return {"status": "already_done", "path": str(cache.root)}

    # Load chunk texts
    chunk_ids, texts = [], []
    with chunks_path.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            chunk_ids.append(rec["chunk_id"])
            texts.append(rec["text_normalized"])
    if not texts:
        return {"status": "error", "reason": "zero chunks"}

    t0 = time.time()
    # Encode in batches for memory safety
    all_dense, all_sparse = [], []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        d, s = embedder.encode_hybrid(batch)
        all_dense.append(d)
        all_sparse.extend(s)
    import numpy as np
    dense = np.concatenate(all_dense, axis=0)
    elapsed = time.time() - t0

    cache.save(
        chunk_ids=chunk_ids,
        dense=dense,
        sparse=all_sparse,
        config={"variant": variant, "dataset": dataset,
                "n_chunks": len(chunk_ids), "elapsed_s": round(elapsed, 2),
                "embedder": "BAAI/bge-m3", "device": embedder.device,
                "max_length": embedder.max_length},
    )
    return {"status": "done", "n_chunks": len(chunk_ids),
            "elapsed_s": round(elapsed, 2),
            "throughput_chunks_per_s": round(len(chunk_ids) / max(elapsed, 0.01), 2)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default=None)
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--tier", type=int, choices=[1, 2, 3], default=None)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)

    jobs = build_plan()
    # Filter to jobs with own embeddings (skip reuse A4)
    jobs = [j for j in jobs if VARIANT_SPECS.get(j["variant"], {}).get("embed_source") == j["variant"]]
    if args.tier:
        jobs = [j for j in jobs if j["tier"] == args.tier]
    if args.variant:
        jobs = [j for j in jobs if j["variant"] == args.variant]
    if args.dataset:
        jobs = [j for j in jobs if j["dataset"] == args.dataset]
    if args.limit:
        jobs = jobs[:args.limit]

    print(f"Encoding {len(jobs)} (variant, dataset) pairs")
    embedder = BGEM3Embedder(max_length=512, use_fp16=True, device="auto", batch_size=args.batch_size)
    print(f"  embedder loaded in {embedder.load_time_s:.1f}s  device={embedder.device}")
    print()

    counts = {}
    t0 = time.time()
    for i, j in enumerate(jobs, 1):
        res = encode_one(j["variant"], j["dataset"], embedder, args.batch_size, args.force)
        s = res.get("status", "?")
        counts[s] = counts.get(s, 0) + 1
        marker = {"done": "+", "already_done": "=", "skipped": "-", "error": "!"}.get(s, "?")
        extra = ""
        if s == "done":
            extra = f"{res['n_chunks']:>7} chunks, {res['elapsed_s']:>6.1f}s, {res['throughput_chunks_per_s']:>6.1f} ch/s"
        elif s == "skipped":
            extra = res.get("reason", "")[:60]
        print(f"[{i:>3}/{len(jobs)}] {marker} {j['variant']:<24} {j['dataset']:<20}  {extra}")

    print()
    print(f"Total: {time.time()-t0:.1f}s  {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())