"""Build BM25 indexes for all datasets (uses A4 chunk source). CLI."""
import argparse
import json
import time

from ablation.config import ALL_DATASETS, CHUNKS_DIR
from ablation.indexers.bm25 import BM25Indexer


def build_one(dataset: str, force: bool = False) -> dict:
    idx = BM25Indexer(variant="A4_full_sah", dataset=dataset)
    if idx.exists() and not force:
        return {"status": "already_done", "path": str(idx.root)}

    chunks_path = CHUNKS_DIR / "A4_full_sah" / dataset / "chunks.jsonl"
    if not chunks_path.is_file():
        return {"status": "skipped", "reason": f"no chunks at {chunks_path}"}

    chunk_ids, texts = [], []
    with chunks_path.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            chunk_ids.append(rec["chunk_id"])
            texts.append(rec["text_normalized"])

    idx.build_and_save(chunk_ids, texts)
    return {"status": "done", "n_chunks": len(chunk_ids)}


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

    print(f"Building BM25 for {len(datasets)} datasets (A4 chunk source)")
    counts = {}
    t0 = time.time()
    for i, ds in enumerate(datasets, 1):
        res = build_one(ds, args.force)
        s = res.get("status", "?")
        counts[s] = counts.get(s, 0) + 1
        marker = {"done": "+", "already_done": "=", "skipped": "-", "error": "!"}.get(s, "?")
        extra = f"{res.get('n_chunks','')} chunks" if s == "done" else res.get("reason", "")[:60]
        print(f"[{i:>2}/{len(datasets)}] {marker} {ds:<22}  {extra}")
    print()
    print(f"Total: {time.time()-t0:.1f}s  {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())