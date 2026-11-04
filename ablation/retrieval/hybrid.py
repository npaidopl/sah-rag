"""Top-level hybrid retrieval pipeline."""
import numpy as np

from ablation.retrieval.fusion import (
    rrf_fuse, weighted_score_fuse, normalized_score_fuse,
)
from ablation.retrieval.aggregation import (
    aggregate_to_parents, resolve_to_documents,
)
from ablation.indexers.sparse_random import randomize_sparse_batch


TOPK_PER_MODE = 200


def _dense_topk(q: np.ndarray, doc_dense: np.ndarray, k: int = TOPK_PER_MODE):
    scores = doc_dense @ q  # dense are L2-normalized → cosine
    order = np.argsort(-scores, kind="stable")[:k]
    return [(int(i), float(scores[i])) for i in order]


def _sparse_dot(q_sparse: dict[int, float], doc_sparse: dict[int, float]) -> float:
    if len(q_sparse) > len(doc_sparse):
        q_sparse, doc_sparse = doc_sparse, q_sparse
    return sum(v * doc_sparse.get(idx, 0.0) for idx, v in q_sparse.items())


def _sparse_topk(q_sparse: dict[int, float], docs_sparse: list[dict], k: int = TOPK_PER_MODE):
    scores = np.array([_sparse_dot(q_sparse, ds) for ds in docs_sparse])
    order = np.argsort(-scores, kind="stable")[:k]
    return [(int(i), float(scores[i])) for i in order]


def _bm25_topk(bm25, tokens: list[str], k: int = TOPK_PER_MODE):
    scores = bm25.get_scores(tokens)
    order = np.argsort(-scores, kind="stable")[:k]
    return [(int(i), float(scores[i])) for i in order]


def retrieve(
    query_dense: np.ndarray | None,
    query_sparse: dict[int, float] | None,
    query_bm25_tokens: list[str] | None,
    doc_dense: np.ndarray | None,
    doc_sparse: list[dict] | None,
    bm25_index,
    chunk_to_parent: list[str],
    parent_to_doc: dict[str, str],
    spec: dict,
) -> list[tuple[str, float, str]]:
    """Run retrieval + fusion + aggregation + resolution. Returns [(doc_id, score, best_parent_id)]."""

    dense_ranked, sparse_ranked, bm25_ranked = [], [], []

    if spec.get("dense") and query_dense is not None:
        dense_ranked = _dense_topk(query_dense, doc_dense)

    if spec.get("sparse") and query_sparse is not None:
        sparse_src = doc_sparse
        if spec.get("sparse_transform") == "randomize":
            # G8 — but the randomization must use chunk_id, not index
            # For now approximate: use index-based seed. Refined later.
            from ablation.indexers.sparse_random import randomize_sparse
            sparse_src = [randomize_sparse(s, f"idx:{i}")
                          for i, s in enumerate(doc_sparse)]
        sparse_ranked = _sparse_topk(query_sparse, sparse_src)

    if spec.get("bm25") and query_bm25_tokens is not None and bm25_index is not None:
        bm25_ranked = _bm25_topk(bm25_index, query_bm25_tokens)

    fusion = spec["fusion"]

    if fusion == "dense_only":
        fused = dense_ranked
    elif fusion == "sparse_only":
        fused = sparse_ranked
    elif fusion == "bm25_only":
        fused = bm25_ranked
    elif fusion == "rrf":
        # For RRF we need rankings as index lists
        dr = [i for i, _ in dense_ranked]
        sr = [i for i, _ in sparse_ranked]
        br = [i for i, _ in bm25_ranked]
        if spec.get("bm25") and dense_ranked and bm25_ranked:
            # D2: two-way RRF (dense + bm25)
            fused = rrf_fuse(dr, br, k=spec.get("rrf_k", 60))
        elif dr and sr:
            fused = rrf_fuse(dr, sr, k=spec.get("rrf_k", 60))
        elif dr:
            fused = dense_ranked
        elif sr:
            fused = sparse_ranked
        else:
            fused = []
    elif fusion == "weighted":
        if dense_ranked and sparse_ranked:
            d_scores = np.zeros(len(doc_dense), dtype=np.float64)
            s_scores = np.zeros(len(doc_dense), dtype=np.float64)
            for i, s in dense_ranked:
                d_scores[i] = s
            for i, s in sparse_ranked:
                s_scores[i] = s
            fused = weighted_score_fuse(d_scores, s_scores,
                                        spec.get("w_dense", 0.5),
                                        spec.get("w_sparse", 0.5))
        else:
            fused = dense_ranked or sparse_ranked
    elif fusion == "normalized":
        if dense_ranked and sparse_ranked:
            d_scores = np.zeros(len(doc_dense), dtype=np.float64)
            s_scores = np.zeros(len(doc_dense), dtype=np.float64)
            for i, s in dense_ranked:
                d_scores[i] = s
            for i, s in sparse_ranked:
                s_scores[i] = s
            fused = normalized_score_fuse(d_scores, s_scores)
        else:
            fused = dense_ranked or sparse_ranked
    else:
        raise ValueError(f"Unknown fusion: {fusion}")

    agg_strategy = spec["aggregation"]
    margin = spec.get("margin", 0.05)
    ranked_parents = aggregate_to_parents(fused, chunk_to_parent, agg_strategy, margin)
    ranked_docs = resolve_to_documents(ranked_parents, parent_to_doc)
    return ranked_docs