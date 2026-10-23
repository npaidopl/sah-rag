"""Controlled-adaptation smoke test for BM25 — with REAL BGE-M3.

Proves the controlled-mode protocol end-to-end:
    1. BGE-M3 is real and functional (dense encoding verified).
    2. BM25 accepts the injected embedder via the frozen contract.
    3. BM25's retrieval output is IDENTICAL to native mode.
    4. BM25 NEVER calls embedder.encode_* (the structural exception).

This is the proof that the "controlled-adaptation" protocol works for BM25:
same corpus, same queries, same shared embedder injected — output identical.

Uses a 30-doc, 5-query slice of SciFact from disk. Metrics are NOT meaningful
(sample too small); this is a protocol integrity check.
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

# Paths
BGE_M3_PATH = Path(r"D:\rag_work\hf_cache\models--BAAI--bge-m3\snapshots\5617a9f61b028005a4858fdac845db406aefb181")
SCIFACT_ROOT = Path(r"D:\rag_work\_scifact\scifact")
WORK_ROOT = Path(r"D:\rag_work\_bm25_smoke_ctrl")
N_DOCS = 30
N_QUERIES = 5
TOP_K = 10


class RealBGE_M3_Embedder:
    """Real Embedder Protocol impl wrapping BGEM3FlagModel.

    Counts encode_* invocations so we can prove BM25 never calls them.
    """
    def __init__(self, model_path: Path):
        print(f"  Loading BGE-M3 from: {model_path}", flush=True)
        t0 = time.time()
        from FlagEmbedding import BGEM3FlagModel
        self._model = BGEM3FlagModel(str(model_path), use_fp16=False)
        self.name = "BAAI/bge-m3 (real, local)"
        self.encode_documents_calls = 0
        self.encode_query_calls = 0
        print(f"  Loaded in {time.time()-t0:.1f}s", flush=True)

    def encode_documents(self, texts):
        self.encode_documents_calls += 1
        out = self._model.encode(texts, return_dense=True, return_sparse=False, return_colbert_vecs=False)
        return out["dense_vecs"].tolist()

    def encode_query(self, text):
        self.encode_query_calls += 1
        out = self._model.encode([text], return_dense=True, return_sparse=False, return_colbert_vecs=False)
        return out["dense_vecs"][0].tolist()

    @property
    def total_encode_calls(self) -> int:
        return self.encode_documents_calls + self.encode_query_calls


def load_slice():
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
                doc_id=str(rec["_id"]), dataset="scifact_slice",
                text_raw=combined, text_normalized=combined,
            ))
    queries = []
    with (SCIFACT_ROOT / "queries.jsonl").open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= N_QUERIES:
                break
            rec = json.loads(line)
            queries.append((str(rec["_id"]), rec.get("text") or rec.get("title") or ""))
    return docs, queries


def main() -> int:
    # Preflight
    if not BGE_M3_PATH.exists():
        print(f"ERROR: BGE-M3 not found at {BGE_M3_PATH}", file=sys.stderr)
        return 1
    if not SCIFACT_ROOT.exists():
        print(f"ERROR: SciFact not found at {SCIFACT_ROOT}", file=sys.stderr)
        return 1
    if WORK_ROOT.exists():
        shutil.rmtree(WORK_ROOT)
    WORK_ROOT.mkdir(parents=True)

    # ---- Phase 1: Load real BGE-M3 and prove it works ----
    print("[1/6] Loading real BGE-M3 and verifying it encodes...", flush=True)
    embedder = RealBGE_M3_Embedder(BGE_M3_PATH)

    print("  Verifying dense encoding:", flush=True)
    t0 = time.time()
    test_vec = embedder.encode_query("retrieval augmented generation")
    print(f"  encoded in {time.time()-t0:.2f}s")
    print(f"  vector dim: {len(test_vec)} (expected 1024 for BGE-M3)")
    if len(test_vec) != 1024:
        print(f"  WARNING: unexpected dim, expected 1024", file=sys.stderr)
    embedder.encode_query_calls = 0  # reset counter
    print("  BGE-M3 OK and functional.")
    print()

    # ---- Phase 2: Load corpus ----
    print(f"[2/6] Loading {N_DOCS} docs + {N_QUERIES} queries from SciFact...", flush=True)
    docs, queries = load_slice()
    print(f"  docs: {len(docs)}  |  queries: {len(queries)}")
    print()

    # ---- Phase 3: Build two separate indices (native + controlled) ----
    print("[3/6] Building native-mode index...", flush=True)
    t0 = time.time()
    cfg_native = BaselineConfig(name="bm25", mode=RunMode.NATIVE, top_k=TOP_K)
    native_bl = BM25Baseline(config=cfg_native, work_dir=WORK_ROOT / "native")
    native_bl.build_index(docs)
    native_build_s = time.time() - t0
    print(f"  built in {native_build_s:.2f}s")

    print("[4/6] Building controlled-mode index (with BGE-M3 injected)...", flush=True)
    t0 = time.time()
    cfg_ctrl = BaselineConfig(name="bm25", mode=RunMode.CONTROLLED, top_k=TOP_K)
    ctrl_bl = BM25Baseline(
        config=cfg_ctrl,
        work_dir=WORK_ROOT / "controlled",
        embedder=embedder,
    )
    ctrl_bl.build_index(docs)
    ctrl_build_s = time.time() - t0
    print(f"  built in {ctrl_build_s:.2f}s")
    print()

    # ---- Phase 5: Retrieve both ----
    print(f"[5/6] Retrieving top-{TOP_K} for {N_QUERIES} queries (native + controlled)...", flush=True)
    native_res = {}
    ctrl_res = {}
    for qid, qtext in queries:
        native_res[qid] = native_bl.retrieve(qtext, top_k=TOP_K)
        ctrl_res[qid] = ctrl_bl.retrieve(qtext, top_k=TOP_K)
    print(f"  done.")
    print()

    # ---- Phase 6: Verify ----
    print("[6/6] Verifying controlled-adaptation protocol...", flush=True)
    failures = []

    # Check 1: output identical (docid + rank + score)
    for qid, qtext in queries:
        n = native_res[qid]
        c = ctrl_res[qid]
        if len(n) != len(c):
            failures.append(f"qid={qid}: length {len(n)} vs {len(c)}")
            continue
        if [r.doc_id for r in n] != [r.doc_id for r in c]:
            failures.append(f"qid={qid}: doc_ids differ")
        if [r.rank for r in n] != [r.rank for r in c]:
            failures.append(f"qid={qid}: ranks differ")
        if [round(r.score, 6) for r in n] != [round(r.score, 6) for r in c]:
            failures.append(f"qid={qid}: scores differ")

    print(f"  Native == Controlled output: {'PASS' if not failures else 'FAIL'}")
    for f in failures:
        print(f"    {f}")

    # Check 2: embedder never called by BM25
    print(f"  Embedder total encode calls: {embedder.total_encode_calls} (expect 0)")
    if embedder.total_encode_calls != 0:
        print(f"    FAIL: BM25 called embedder {embedder.total_encode_calls} times!")
        failures.append(f"embedder was invoked ({embedder.total_encode_calls} calls)")
    else:
        print(f"    PASS: embedder structurally unused (as expected for BM25)")

    # Cleanup
    native_bl.cleanup()
    ctrl_bl.cleanup()

    print()
    if failures:
        print("=" * 60, file=sys.stderr)
        print("CONTROLLED SMOKE FAILED", file=sys.stderr)
        for f in failures:
            print(f"  {f}", file=sys.stderr)
        return 1

    print("=" * 60)
    print("CONTROLLED-ADAPTATION SMOKE PASSED")
    print(f"  BGE-M3 loaded from: {BGE_M3_PATH.name[:16]}...")
    print(f"  Native == Controlled:  YES (identical docids, ranks, scores)")
    print(f"  Embedder calls by BM25: 0  (structural exception confirmed)")
    print(f"  Protocol: shared embedder injected, accepted, structurally unused")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())