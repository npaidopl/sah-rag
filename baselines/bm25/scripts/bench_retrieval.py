"""Benchmark: separate JVM init from steady-state retrieval.

Runs N queries against the full SciFact corpus (if available) or a slice.
Reports: JVM init time, per-query time (raw), per-query time (steady state).
"""
from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from baselines.base import BaselineConfig, RunMode
from baselines.bm25.adapter import BM25Baseline
from sah_rag.canonical.models import CanonicalDocument

SCIFACT_ROOT = Path(r"D:\rag_work\_scifact\scifact")
WORK_DIR = Path(r"D:\rag_work\_bm25_bench_idx")
N_DOCS = 500
N_QUERIES = 50
TOP_K = 10


def main() -> int:
    if not SCIFACT_ROOT.exists():
        print(f"ERROR: SciFact not found at {SCIFACT_ROOT}", file=sys.stderr)
        return 1
    if WORK_DIR.exists():
        shutil.rmtree(WORK_DIR)

    # Load docs + queries
    docs = []
    with (SCIFACT_ROOT / "corpus.jsonl").open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= N_DOCS:
                break
            rec = json.loads(line)
            title = (rec.get("title") or "").strip()
            text = (rec.get("text") or "").strip()
            combined = f"{title} {text}".strip()
            docs.append(CanonicalDocument(
                doc_id=str(rec["_id"]), dataset="scifact_bench",
                text_raw=combined, text_normalized=combined,
            ))
    queries = []
    with (SCIFACT_ROOT / "queries.jsonl").open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= N_QUERIES:
                break
            rec = json.loads(line)
            queries.append(rec.get("text") or rec.get("title") or "")

    print(f"docs={len(docs)}  queries={len(queries)}  top_k={TOP_K}")
    print()

    # Build
    cfg = BaselineConfig(name="bm25", mode=RunMode.NATIVE, top_k=TOP_K)
    baseline = BM25Baseline(config=cfg, work_dir=WORK_DIR)
    t0 = time.perf_counter()
    baseline.build_index(docs)
    build_s = time.perf_counter() - t0
    print(f"[build_index]  {build_s:.2f}s")
    print()

    # First query includes JVM init
    t0 = time.perf_counter()
    baseline.retrieve(queries[0], top_k=TOP_K)
    jvm_s = time.perf_counter() - t0
    print(f"[first query / JVM init]  {jvm_s:.2f}s")
    print()

    # Remaining queries (steady state candidates)
    timings = []
    for q in queries[1:]:
        t0 = time.perf_counter()
        baseline.retrieve(q, top_k=TOP_K)
        timings.append(time.perf_counter() - t0)

    # Stats
    total_s = sum(timings)
    mean_ms = (total_s / len(timings)) * 1000
    median_ms = sorted(timings)[len(timings)//2] * 1000
    p95_ms = sorted(timings)[int(len(timings)*0.95)] * 1000
    warm_10 = timings[9:]  # skip first 10 for "warm" measurement

    print(f"[per-query stats, n={len(timings)}]")
    print(f"  mean:    {mean_ms:.1f} ms")
    print(f"  median:  {median_ms:.1f} ms")
    print(f"  p95:     {p95_ms:.1f} ms")
    if warm_10:
        warm_ms = (sum(warm_10)/len(warm_10)) * 1000
        print(f"  warm (skipping first 10): {warm_ms:.1f} ms")
    print()

    # Steady-state rate
    warm_mean_s = sum(warm_10)/len(warm_10) if warm_10 else (total_s/len(timings))
    print(f"Steady-state rate: {1/warm_mean_s:.1f} q/s")
    print(f"Projected for 300 queries (SciFact full):   {300*warm_mean_s:.1f}s")
    print(f"Projected for 6,980 queries (MS MARCO):     {6980*warm_mean_s/60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())