"""Self-contained retrieval metrics for SAH-RAG v2.

Metric version: v2 (CUTOFFS=[1,5,10], hit@k added).
See docs/evaluation_boundary.md for full protocol.
"""
from __future__ import annotations

import math


def recall_at_k(retrieved_doc_ids: list[str], gt_doc_ids: list[str], k: int) -> float:
    if not gt_doc_ids or k <= 0:
        return 0.0
    top = set(retrieved_doc_ids[:k])
    return len(top & set(gt_doc_ids)) / len(set(gt_doc_ids))


def precision_at_k(retrieved_doc_ids: list[str], gt_doc_ids: list[str], k: int) -> float:
    if k <= 0:
        return 0.0
    top = set(retrieved_doc_ids[:k])
    return len(top & set(gt_doc_ids)) / k


def mrr_at_k(retrieved_doc_ids: list[str], gt_doc_ids: list[str], k: int) -> float:
    gt = set(gt_doc_ids)
    for rank, doc in enumerate(retrieved_doc_ids[:k], start=1):
        if doc in gt:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_doc_ids: list[str], gt_doc_ids: list[str], k: int) -> float:
    gt = set(gt_doc_ids)
    if not gt or k <= 0:
        return 0.0
    dcg = sum(1.0 / math.log2(rank + 1) for rank, doc in enumerate(retrieved_doc_ids[:k], start=1) if doc in gt)
    ideal_len = min(k, len(gt))
    idcg = sum(1.0 / math.log2(r + 1) for r in range(1, ideal_len + 1))
    return dcg / idcg if idcg > 0 else 0.0


def hit_at_k(retrieved_doc_ids: list[str], gt_doc_ids: list[str], k: int) -> float:
    """Binary hit: 1.0 if any gt doc appears in top-k retrieved, else 0.0."""
    if not gt_doc_ids or k <= 0:
        return 0.0
    gt = set(gt_doc_ids)
    return 1.0 if any(d in gt for d in retrieved_doc_ids[:k]) else 0.0


__all__ = ["recall_at_k", "precision_at_k", "mrr_at_k", "ndcg_at_k", "hit_at_k"]