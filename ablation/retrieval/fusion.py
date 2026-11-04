"""Score fusion: RRF (k configurable) and weighted/normalized score fusion."""
import numpy as np


def rrf_fuse(dense_ranked: list[int], sparse_ranked: list[int],
             k: int = 60, top_n: int | None = None) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion. Returns [(idx, score), ...] sorted desc."""
    scores: dict[int, float] = {}
    for r, idx in enumerate(dense_ranked, 1):
        scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + r)
    for r, idx in enumerate(sparse_ranked, 1):
        scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + r)
    out = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    return out[:top_n] if top_n else out


def _minmax(arr: np.ndarray) -> np.ndarray:
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-12:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def weighted_score_fuse(dense_scores: np.ndarray, sparse_scores: np.ndarray,
                        w_dense: float = 0.5, w_sparse: float = 0.5,
                        top_n: int | None = None) -> list[tuple[int, float]]:
    """Min-max normalize each score array, weight-sum, rank."""
    d = _minmax(dense_scores)
    s = _minmax(sparse_scores)
    fused = w_dense * d + w_sparse * s
    order = np.argsort(-fused, kind="stable")
    if top_n:
        order = order[:top_n]
    return [(int(i), float(fused[i])) for i in order]


def normalized_score_fuse(dense_scores: np.ndarray, sparse_scores: np.ndarray,
                          top_n: int | None = None) -> list[tuple[int, float]]:
    """Score fusion after L2 normalization of each score vector."""
    dn = dense_scores / (np.linalg.norm(dense_scores) + 1e-12)
    sn = sparse_scores / (np.linalg.norm(sparse_scores) + 1e-12)
    fused = dn + sn
    order = np.argsort(-fused, kind="stable")
    if top_n:
        order = order[:top_n]
    return [(int(i), float(fused[i])) for i in order]