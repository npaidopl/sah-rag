"""Encode queries for all datasets. CLI entry.

Usage:
  python -m ablation.embedding.encode_queries --dataset clerk
  python -m ablation.embedding.encode_queries --all
"""
import argparse
import time

from ablation.config import ALL_DATASETS
from ablation.embedding.bge_m3 import BGEM3Embedder
from ablation.embedding.query_cache import QueryEmbeddingCache
from ablation.loaders import iter_canonical_examples_local


def encode_one(dataset: str, embedder: BGEM3Embedder, force: bool = False) -> dict:
    cache = QueryEmbeddingCache(dataset)
    if cache.exists() and not force:
        return {"status": "already_done", "path": str(cache.root)}

    qids, texts = [], []
    for ex in iter_canonical_examples_local(dataset):
        qids.append(ex.query.query_id)
        texts.append(ex.query.question)
    if not texts:
        return {"status": "skipped", "reason": "no queries"}

    t0 = time.time()
    dense, sparse = embedder.encode_hybrid(texts)
    elapsed = time.time() - t0
    cache.save(qids, dense, sparse, config={
        "dataset": dataset, "n": len(qids), "elapsed_s": round(elapsed, 2),
        "embedder": "BAAI/bge-m3", "device": embedder.device,
        "max_length": embedder.max_length,
    })
    return {"status": "done", "n": len(qids), "elapsed_s": round(elapsed, 2)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    datasets = [args.dataset] if args.dataset else (ALL_DATASETS if args.all else [])
    if not datasets:
        print("Specify --dataset X or --all")
        return 1

    print(f"Encoding queries for {len(datasets)} datasets")
    embedder = BGEM3Embedder(max_length=512, use_fp16=True, device="auto")
    print(f"  embedder loaded in {embedder.load_time_s:.1f}s  device={embedder.device}")
    print()

    counts = {}
    t0 = time.time()
    for i, ds in enumerate(datasets, 1):
        res = encode_one(ds, embedder, args.force)
        s = res.get("status", "?")
        counts[s] = counts.get(s, 0) + 1
        marker = {"done": "+", "already_done": "=", "skipped": "-", "error": "!"}.get(s, "?")
        extra = f"{res.get('n','')} queries, {res.get('elapsed_s','')}s" if s == "done" else res.get("reason", "")
        print(f"[{i:>2}/{len(datasets)}] {marker} {ds:<22}  {extra}")

    print()
    print(f"Total: {time.time()-t0:.1f}s  {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())