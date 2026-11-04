"""Job executor: load chunks+embeddings+BM25, retrieve, evaluate, save. Resumable."""
import hashlib
import json
import time
import traceback
from pathlib import Path

import numpy as np

from ablation.config import RESULTS_DIR, CHUNKS_DIR
from ablation.loaders import iter_canonical_examples_local
from ablation.embedding.cache import EmbeddingCache
from ablation.embedding.query_cache import QueryEmbeddingCache
from ablation.indexers.bm25 import BM25Indexer, tokenize
from ablation.retrieval.hybrid import retrieve
from ablation.retrieval.variant_specs import VARIANT_SPECS
from evaluation.retrieval_metrics import (
    recall_at_k, precision_at_k, mrr_at_k, ndcg_at_k, hit_at_k,
)

CUTOFFS = (10, 50)


def _result_paths(variant: str, dataset: str):
    base = RESULTS_DIR / variant / dataset
    base.mkdir(parents=True, exist_ok=True)
    return {"root": base,
            "per_query": base / "per_query_metrics.jsonl",
            "summary": base / "summary.json",
            "run_manifest": base / "run_manifest.json"}


def _load_chunk_metadata(variant: str, dataset: str):
    """Load chunk_id, parent_id, doc_id, text from chunks cache."""
    p = CHUNKS_DIR / variant / dataset / "chunks.jsonl"
    if not p.is_file():
        return None
    chunk_ids, parent_ids, doc_ids, texts = [], [], [], []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            chunk_ids.append(rec["chunk_id"])
            parent_ids.append(rec["parent_id"])
            doc_ids.append(rec["doc_id"])
            texts.append(rec["text_normalized"])
    return {"chunk_ids": chunk_ids, "parent_ids": parent_ids,
            "doc_ids": doc_ids, "texts": texts}


def _mock_dense_from_id(sid: str) -> np.ndarray:
    h = hashlib.sha256(sid.encode()).digest()
    rng = np.random.default_rng(int.from_bytes(h[:8], "big"))
    v = rng.standard_normal(1024).astype(np.float32)
    return v / np.linalg.norm(v)


def _mock_sparse_from_id(sid: str) -> dict[int, float]:
    h = hashlib.sha256(sid.encode()).digest()
    rng = np.random.default_rng(int.from_bytes(h[:8], "big"))
    idxs = rng.choice(250_000, size=100, replace=False)
    vals = rng.uniform(0.05, 1.0, size=100)
    return {int(i): float(v) for i, v in zip(idxs, vals)}


def run_job(job: dict, dry_run: bool = False, mock_embeddings: bool = False) -> dict:
    """Execute one (variant, dataset) cell. Idempotent unless dry_run."""
    vname = job["variant"]
    dataset = job["dataset"]
    spec = VARIANT_SPECS.get(vname)
    if spec is None:
        return {"status": "skipped", "reason": f"unknown variant {vname}"}
    if job.get("status") == "placeholder":
        return {"status": "skipped", "reason": "placeholder (tier 3)"}

    paths = _result_paths(vname, dataset)
    if paths["summary"].exists() and not dry_run:
        return {"status": "already_done", "path": str(paths["summary"])}

    t0 = time.time()
    try:
        meta = _load_chunk_metadata(spec["chunk_source"], dataset)
        if meta is None:
            return {"status": "skipped",
                    "reason": f"no chunks at {CHUNKS_DIR / spec['chunk_source'] / dataset}"}

        chunk_ids = meta["chunk_ids"]
        parent_ids = meta["parent_ids"]
        doc_ids = meta["doc_ids"]
        texts = meta["texts"]
        pid2doc = dict(zip(parent_ids, doc_ids))

        # ---- Load embeddings or mock ----
        doc_dense, doc_sparse = None, None
        if mock_embeddings:
            if spec.get("dense"):
                doc_dense = np.stack([_mock_dense_from_id(c) for c in chunk_ids])
            if spec.get("sparse"):
                doc_sparse = [_mock_sparse_from_id(c) for c in chunk_ids]
        else:
            if spec.get("dense") or spec.get("sparse"):
                ec = EmbeddingCache(variant=spec["embed_source"], dataset=dataset)
                if not ec.exists():
                    return {"status": "skipped", "reason": f"no embed cache: {ec.root}"}
                cids_e, doc_dense_full, doc_sparse_full = ec.load()
                if cids_e != chunk_ids:
                    return {"status": "error",
                            "reason": f"chunk_id mismatch: cache={len(cids_e)} meta={len(chunk_ids)}"}
                if spec.get("dense"):
                    doc_dense = doc_dense_full
                if spec.get("sparse"):
                    doc_sparse = doc_sparse_full

        # ---- Load BM25 ----
        bm25_index = None
        if spec.get("bm25"):
            bi = BM25Indexer(variant=spec["chunk_source"], dataset=dataset)
            if not bi.exists():
                return {"status": "skipped", "reason": f"no BM25: {bi.root}"}
            cids_b, bm25_index = bi.load()
            if cids_b != chunk_ids:
                return {"status": "error", "reason": "BM25 chunk_id mismatch"}

        # ---- Load queries ----
        examples = list(iter_canonical_examples_local(dataset))
        if not examples:
            return {"status": "error", "reason": "no examples"}
        if dry_run:
            examples = examples[:5]

        # ---- Query embeddings ----
        real_q_cache = None
        if not mock_embeddings:
            try:
                real_q_cache = QueryEmbeddingCache(dataset).load()
            except FileNotFoundError:
                return {"status": "skipped",
                        "reason": f"no query embedding cache for {dataset}"}

        # ---- Retrieval loop ----
        per_query = []
        for ex in examples:
            qid = ex.query.query_id
            gt = ex.ground_truth.relevant_ids
            if mock_embeddings or real_q_cache is None:
                q_dense = _mock_dense_from_id(qid) if spec.get("dense") else None
                q_sparse = _mock_sparse_from_id(qid) if spec.get("sparse") else None
            else:
                d, s = real_q_cache[qid]
                q_dense = d if spec.get("dense") else None
                q_sparse = s if spec.get("sparse") else None
            q_bm25_tokens = tokenize(ex.query.question)

            ranked = retrieve(
                query_dense=q_dense, query_sparse=q_sparse,
                query_bm25_tokens=q_bm25_tokens,
                doc_dense=doc_dense, doc_sparse=doc_sparse,
                bm25_index=bm25_index,
                chunk_to_parent=parent_ids, parent_to_doc=pid2doc,
                spec=spec,
            )
            retrieved_ids = [d[0] for d in ranked]
            row = {"query_id": qid, "n_gt": len(gt),
                   "n_retrieved": len(retrieved_ids),
                   "top10_ids": retrieved_ids[:10]}
            for k in CUTOFFS:
                row[f"recall@{k}"]    = recall_at_k(retrieved_ids, gt, k)
                row[f"precision@{k}"] = precision_at_k(retrieved_ids, gt, k)
                row[f"mrr@{k}"]       = mrr_at_k(retrieved_ids, gt, k)
                row[f"ndcg@{k}"]      = ndcg_at_k(retrieved_ids, gt, k)
                row[f"hit@{k}"]       = hit_at_k(retrieved_ids, gt, k)
            per_query.append(row)

        n = len(per_query)
        summary = {"variant": vname, "dataset": dataset, "n_queries": n,
                   "elapsed_s": round(time.time() - t0, 2)}
        for k in CUTOFFS:
            for m in ("recall", "precision", "mrr", "ndcg", "hit"):
                key = f"{m}@{k}"
                summary[key] = round(sum(r[key] for r in per_query) / n, 6)

        with paths["per_query"].open("w", encoding="utf-8") as f:
            for r in per_query:
                f.write(json.dumps(r) + "\n")
        paths["summary"].write_text(json.dumps(summary, indent=2), encoding="utf-8")
        paths["run_manifest"].write_text(json.dumps({
            "variant": vname, "dataset": dataset, "spec": spec,
            "n_queries": n, "cutoffs": list(CUTOFFS),
            "dry_run": dry_run, "mock_embeddings": mock_embeddings,
            "elapsed_s": summary["elapsed_s"],
        }, indent=2), encoding="utf-8")
        return {"status": "done", "summary_path": str(paths["summary"]),
                "elapsed_s": summary["elapsed_s"], "n_queries": n}
    except Exception:
        return {"status": "error", "traceback": traceback.format_exc()}