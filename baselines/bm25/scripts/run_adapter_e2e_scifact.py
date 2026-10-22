"""Adapter-level SciFact E2E: build our own index, retrieve, evaluate.

This is a COMPLEMENT to Stage 1 (which used the official Pyserini pre-built
index and reproduced 0.6789 nDCG@10). This script runs OUR adapter, which
builds its own Lucene index from CanonicalDocument objects. It will NOT
reproduce 0.6789 exactly (different index build path, potentially different
tokenization), but proves the adapter works end-to-end and lands in a
reasonable range.

Output:
- Run file (TREC format) at baselines/bm25/runs/scifact.adapter_e2e.trec
- Prints mean nDCG@10 and per-some-query timing

Usage:
    python run_adapter_e2e_scifact.py
"""
import sys
import time
from pathlib import Path

# Make project root importable
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import pytrec_eval
from datasets import load_dataset
from pyserini.search import get_qrels_file, get_topics

from baselines.base import BaselineConfig, RunMode
from sah_rag.canonical.models import CanonicalDocument
from baselines.bm25.adapter import BM25Baseline


def load_qrels(path: Path) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            qid, _, docid, rel = parts[0], parts[1], parts[2], parts[3]
            qrels.setdefault(qid, {})[docid] = int(rel)
    return qrels


def main() -> None:
    print("Loading SciFact corpus from HuggingFace (BeIR/scifact)...")
    corpus = load_dataset("BeIR/scifact", "corpus", split="corpus")
    print(f"  loaded {len(corpus)} documents")

    print("Converting to CanonicalDocument (text_raw = title + ' ' + text)...")
    docs = []
    for row in corpus:
        title = (row.get("title") or "").strip()
        text = (row.get("text") or "").strip()
        combined = f"{title} {text}".strip()
        docs.append(CanonicalDocument(
            doc_id=str(row["_id"]),
            dataset="scifact",
            text_raw=combined,
            text_normalized=combined,
            title=title or None,
        ))
    print(f"  built {len(docs)} CanonicalDocument")

    print("Building adapter index (own Lucene index)...")
    t0 = time.time()
    cfg = BaselineConfig(name="bm25", mode=RunMode.NATIVE, top_k=10)
    work_dir = Path(__file__).resolve().parents[1] / "runs" / "_e2e_index"
    baseline = BM25Baseline(config=cfg, work_dir=work_dir)
    baseline.build_index(docs)
    print(f"  index built in {time.time()-t0:.1f}s -> {work_dir}")

    print("Loading SciFact queries...")
    topics = get_topics("beir-v1.0.0-scifact-test")
    print(f"  loaded {len(topics)} queries")

    print("Retrieving (this takes ~2-4 min for 300 queries)...")
    run_file = Path(__file__).resolve().parents[1] / "runs" / "scifact.adapter_e2e.trec"
    t0 = time.time()
    n_q = 0
    with run_file.open("w", encoding="utf-8") as f:
        for qid, q in topics.items():
            query_text = q.get("title", "")
            results = baseline.retrieve(query_text, top_k=10)
            for r in results:
                # TREC format: qid Q0 docid rank score tag
                f.write(f"{qid} Q0 {r.doc_id} {r.rank} {r.score:.6f} bm25_adapter\n")
            n_q += 1
            if n_q % 25 == 0:
                elapsed = time.time() - t0
                rate = n_q / elapsed if elapsed > 0 else 0
                print(f"  [{n_q}/{len(topics)}] {rate:.1f} q/s")
    elapsed = time.time() - t0
    print(f"  retrieved {n_q} queries in {elapsed:.1f}s ({n_q/elapsed:.1f} q/s)")

    print("Evaluating nDCG@10 with pytrec_eval...")
    qrels_path = Path(get_qrels_file("beir-v1.0.0-scifact-test"))
    qrels = load_qrels(qrels_path)

    # Load our run file back
    run: dict[str, dict[str, float]] = {}
    with run_file.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            qid, _, docid, _, score, _ = parts
            run.setdefault(qid, {})[docid] = float(score)

    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {"ndcg_cut_10"})
    scores = evaluator.evaluate(run)
    values = [s["ndcg_cut_10"] for s in scores.values()]
    mean = sum(values) / len(values)

    print()
    print(f"ADAPTER_SCIFACT_E2E nDCG@10 = {mean:.6f}")
    print(f"queries evaluated: {len(values)}")
    print(f"run file: {run_file}")


if __name__ == "__main__":
    main()