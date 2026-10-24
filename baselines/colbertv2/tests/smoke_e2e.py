r"""Standalone native end-to-end smoke test for the ColBERTv2 baseline.

Uses .venv-colbert interpreter. On Windows (no MSVC), applies pure-Python
fallback for C++ extensions so the pipeline can run for verification.
On Linux, the fallback is unnecessary (native C++ compiles).

Loads real SciFact data (BEIR benchmark):
  - 100 documents from corpus.jsonl
  - 10 queries from queries.jsonl

Exit code 0 = success.
"""
from __future__ import annotations

import json
import os

# SAH_DLL_DIRS_PATCHED
# Ensure pthreadVC2.dll is discoverable before colbert's C++ extensions load.
if os.name == "nt":
    _dll_dirs = [
        r"D:\rag_work\.venv-colbert\Scripts",
        r"D:\MSVC\installer\pthreads-extracted\Pre-built.2\dll\x64",
    ]
    for _d in _dll_dirs:
        if os.path.isdir(_d):
            try:
                os.add_dll_directory(_d)
                print(f"[SAH] DLL dir added: {_d}")
            except Exception as _e:
                print(f"[SAH] WARN: add_dll_directory({_d}) failed: {_e}")


import shutil
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---- Apply Windows compatibility patch BEFORE importing colbert anything ----
if os.name == "nt" and os.environ.get("SAH_SKIP_WIN_COMPAT", "0") != "1":
    from baselines.colbertv2._win_compat import patch_colbert_for_windows
    patch_colbert_for_windows()
    print("[SAH] Windows compat patch APPLIED (pure-Python fallback)")
elif os.name == "nt":
    print("[SAH] Windows compat patch SKIPPED - native C++ compile with gcc")

from baselines.base import BaselineConfig, RunMode
from sah_rag.canonical.models import CanonicalDocument
from baselines.colbertv2.adapter import ColBERTv2Baseline

REUSE_INDEX = os.environ.get("SAH_REUSE_INDEX", "1") != "0"


def main() -> int:
    base = Path(r"D:\rag_work\_scifact\scifact")
    work = Path(r"D:\rag_work\_scifact_colbertv2_native_idx")

    if not (base / "corpus.jsonl").exists():
        print(f"ERROR: SciFact corpus not found at {base}", file=sys.stderr)
        return 2

    N_DOCS, N_QUERIES = 100, 10

    docs: list[CanonicalDocument] = []
    with (base / "corpus.jsonl").open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= N_DOCS:
                break
            rec = json.loads(line)
            text = (rec.get("title") or "") + ". " + (rec.get("text") or "")
            docs.append(CanonicalDocument(
                doc_id=rec["_id"],
                dataset="scifact_slice",
                text_raw=text,
                text_normalized=text,
            ))

    queries: list[tuple[str, str]] = []
    with (base / "queries.jsonl").open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= N_QUERIES:
                break
            rec = json.loads(line)
            queries.append((rec["_id"], rec["text"]))

    print(f"Loaded {len(docs)} docs, {len(queries)} queries from SciFact\n")

    cfg = BaselineConfig(
        name="colbertv2",
        mode=RunMode.NATIVE,
        top_k=3,
        extra={
            "checkpoint": "colbert-ir/colbertv2.0",
            "doc_maxlen": 512,
            "nbits": 2,
            "ncells": 1,
            "chunk_size": 1800,
            "chunk_overlap": 200,
            "experiment": "colbertv2_native",
            "index_name": "corpus",
        },
    )

    index_meta = work / "colbert_experiments" / "colbertv2_native" / "indexes" / "corpus" / "metadata.json"
    reuse = REUSE_INDEX and index_meta.exists() and (work / "mapping.json").exists() and (work / "collection.tsv").exists()

    if reuse:
        print(f"Reusing index at {work}")
        bl = ColBERTv2Baseline(config=cfg, work_dir=work)
        bl._built = True
        bl._get_mapping()
        n_chunks = len(bl._mapping) if bl._mapping else -1
        print(f"Reused: {n_chunks} chunks\n")
    else:
        if work.exists():
            shutil.rmtree(work)
        bl = ColBERTv2Baseline(config=cfg, work_dir=work)
        print(f"Building index at {work} (may take 1-3 min) ...")
        bl.build_index(docs)
        n_chunks = len(bl._mapping) if bl._mapping else -1
        print(f"Index built: {n_chunks} chunks from {len(docs)} docs\n")

    print("=== Native ColBERTv2 end-to-end results ===\n")
    total_retrieved = 0
    empty_queries = 0
    for qid, qtext in queries:
        print(f"Query [{qid}]: {qtext}")
        results = bl.retrieve(qtext, top_k=3)
        if not results:
            print("  (no results)\n")
            empty_queries += 1
            continue
        for r in results:
            snippet = r.text[:110].replace("\n", " ")
            print(f"  rank={r.rank} score={r.score:.4f}  doc={r.doc_id}  chunk={r.chunk_id}")
            print(f"    {snippet}...")
        total_retrieved += len(results)
        print()

    assert len(docs) == N_DOCS
    assert (work / "collection.tsv").exists()
    assert (work / "mapping.json").exists()
    assert bl._mapping is not None and len(bl._mapping) >= N_DOCS

    corpus_ids = {d.doc_id for d in docs}
    for qid, qtext in queries:
        res = bl.retrieve(qtext, top_k=3)
        assert [r.rank for r in res] == list(range(1, len(res) + 1)), \
            f"rank not contiguous for query {qid}: {[r.rank for r in res]}"
        for r in res:
            assert r.doc_id in corpus_ids
            assert r.text
            assert isinstance(r.score, float)
            assert r.chunk_id.startswith(r.doc_id + "_c")

    bl.cleanup()

    print("=" * 60)
    print("OK: native ColBERTv2 end-to-end verified on REAL BEIR data.")
    print(f"     - Documents indexed : {len(docs)}")
    print(f"     - Chunks indexed    : {n_chunks}")
    print(f"     - Queries executed  : {len(queries)}")
    print(f"     - Queries with hits : {len(queries) - empty_queries}")
    print(f"     - Queries empty     : {empty_queries}")
    print(f"     - Total retrieved   : {total_retrieved}")
    print(f"     - Index persisted   : {work}")
    print(f"     - Index reused      : {reuse}")
    print(f"     - Mode              : AUTHOR_SUPPORTED_NATIVE (official ckpt)")
    if os.name == "nt":
        print(f"     - C++ ext           : DISABLED (Windows fallback; Linux native)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

