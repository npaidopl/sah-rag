"""Random sparse control: keeps the SUPPORT of the learned sparse but
randomizes the WEIGHTS (same density, same indices, random values).

Deterministic per chunk_id — reproducible across runs.
"""
import hashlib
import random


def randomize_sparse(sparse: dict[int, float], chunk_id: str,
                     value_range: tuple[float, float] = (0.05, 1.0)) -> dict[int, float]:
    if not sparse:
        return {}
    h = hashlib.sha256(f"random_sparse:{chunk_id}".encode("utf-8")).hexdigest()
    rng = random.Random(int(h[:16], 16))
    lo, hi = value_range
    return {idx: rng.uniform(lo, hi) for idx in sparse.keys()}


def randomize_sparse_batch(sparse_batch: list[dict[int, float]],
                           chunk_ids: list[str]) -> list[dict[int, float]]:
    return [randomize_sparse(s, cid) for s, cid in zip(sparse_batch, chunk_ids)]