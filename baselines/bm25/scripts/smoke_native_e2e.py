"""Native E2E smoke test for BM25 — the "drop" dataset.

Purpose: prove the full pipeline (CanonicalDocument -> build_index -> retrieve)
runs end-to-end on real data, using a tiny slice of SciFact (30 docs, 5 queries).
Metrics are NOT meaningful (sample too small); this is a pipeline integrity check.

Not a replacement for the full SciFact/SAH-RAG runs. Complementary to
tests/test_adapter.py (which uses inline mock docs) by using REAL data from
the downloaded SciFact corpus on disk.

Usage:
    python baselines/bm25/scripts/smoke_native_e2e.py
"""
from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

# Make project root importable
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from baselines.base import BaselineConfig, RunMode
from baselines.bm25.adapter import BM25Baseline
from sah_rag.canonical.models import CanonicalDocument

# Config
SCIFACT_ROOT = Path(r"D:\rag_work\_scifact\scifact")
WORK_DIR = Path(r"D:\rag_work\_bm25_smoke_native_idx")
N_DOCS = 30
N_QUERIES = 5
TOP_K = 10


def main() -> int:
    if not SCIFACT_ROOT.exists():
        print(f"ERROR: SciFact not found at {SCIFACT_ROOT}", file=sys.stderr)
        return 1

    if WORK_DIR.exists():
        shutil.rmtree(WORK_DIR)

    # --- Load real data ---
    print(f"[1/5] Loading {N_DOCS} docs + {N_QUERIES} queries from SciFact...", flush=True)
    docs: list[CanonicalDocument] = []
    with (SCIFACT_ROOT / "corpus.jsonl").open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= N_DOCS:
                break
            rec = json.loads(line)
            title = (rec.get("title") or "").strip()
            text = (rec.get("text") or "").strip()
            combined = f"{title} {text}".strip()
            docs.append(CanonicalDocument(
                doc_id=str(rec["_id"]),
                dataset="scifact_slice",
                text_raw=combined,
                text_normalized=combined,
            ))
    print(f"  docs: {len(docs)}")

    queries: list[tuple[str, str]] = []
    with (SCIFACT_ROOT / "queries.jsonl").open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= N_QUERIES:
                break
            rec = json.loads(line)
            queries.append((str(rec["_id"]), rec.get("text") or rec.get("title") or ""))
    print(f"  queries: {len(queries)}")
    for qid, qtext in queries:
        print(f"    {qid}: {qtext[:60]!r}")

    # --- Build index ---
    print(f"\n[2/5] Building BM25 index in {WORK_DIR}...", flush=True)
    t0 = time.time()
    cfg = BaselineConfig(name="bm25", mode=RunMode.NATIVE, top_k=TOP_K)
    baseline = BM25Baseline(config=cfg, work_dir=WORK_DIR)
    baseline.build_index(docs)
    build_s = time.time() - t0
    print(f"  built in {build_s:.1f}s")

    if not (WORK_DIR / "lucene_index").exists():
        print("ERROR: Lucene index dir not created", file=sys.stderr)
        return 1

    # --- Retrieve ---
    print(f"\n[3/5] Retrieving top-{TOP_K} for {N_QUERIES} queries...", flush=True)
    t0 = time.time()
    all_results: dict[str, list] = {}
    for qid, qtext in queries:
        results = baseline.retrieve(qtext, top_k=TOP_K)
        all_results[qid] = results
        print(f"  qid={qid}  n={len(results)}")
    retrieve_s = time.time() - t0
    print(f"  retrieved in {retrieve_s:.2f}s ({N_QUERIES/retrieve_s:.1f} q/s)")

    # --- Validate output ---
    print(f"\n[4/5] Validating output structure...", flush=True)
    corpus_ids = {d.doc_id for d in docs}
    errors = []
    for qid, results in all_results.items():
        if not isinstance(results, list):
            errors.append(f"qid={qid}: not a list")
            continue
        if len(results) > TOP_K:
            errors.append(f"qid={qid}: exceeds top_k ({len(results)} > {TOP_K})")
        ranks = [r.rank for r in results]
        expected = list(range(1, len(results) + 1))
        if ranks != expected:
            errors.append(f"qid={qid}: non-contiguous ranks {ranks}")
        for r in results:
            if r.doc_id not in corpus_ids:
                errors.append(f"qid={qid}: unknown doc_id {r.doc_id}")
            if not r.text:
                errors.append(f"qid={qid}: empty text for {r.doc_id}")
            if r.chunk_id != r.doc_id:
                errors.append(f"qid={qid}: chunk_id != doc_id for BM25")
    if errors:
        print("  VALIDATION FAILED:", file=sys.stderr)
        for e in errors:
            print(f"    {e}", file=sys.stderr)
        return 1
    print("  OK: all results valid")

    # --- Show sample ---
    print(f"\n[5/5] Sample output (qid={queries[0][0]}):", flush=True)
    for r in all_results[queries[0][0]][:5]:
        print(f"  rank={r.rank}  doc={r.doc_id}  score={r.score:.4f}  text={r.text[:60]!r}")

    print()
    print("=" * 60)
    print("SMOKE TEST PASSED — BM25 native E2E pipeline works end-to-end.")
    print(f"  Docs: {len(docs)}  |  Queries: {len(queries)}  |  Index: {WORK_DIR}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())