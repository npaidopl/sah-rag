"""Exact in-memory dense reference backend for LongRAG equivalence proof.

This module provides a deterministic, exact reference implementation of
dense cosine retrieval. It does not use approximate nearest-neighbor
search, quantization, or candidate pruning. It exists only to prove that
the project-owned Qdrant storage path returns the same ranked identities
on the frozen synthetic acceptance fixture.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def _l2_normalize(vec: Sequence[float]) -> tuple[float, ...]:
    norm_sq = 0.0
    for x in vec:
        if not math.isfinite(x):
            raise ValueError("Vector components must be finite.")
        norm_sq += x * x
    norm = math.sqrt(norm_sq)
    if norm == 0.0:
        raise ValueError("Cannot normalize a zero vector.")
    return tuple(float(x) / norm for x in vec)


class ReferenceDenseIndex:
    """Exact in-memory dense index using cosine via normalized dot product.

    Deterministic ordering contract:
        score DESC, long_unit_id ASC
    """

    def __init__(self) -> None:
        self._ids: list[str] = []
        self._vectors: list[tuple[float, ...]] = []
        self._dimension: int | None = None
        self._frozen = False

    def add(self, long_unit_id: str, dense_vector: Sequence[float]) -> None:
        if self._frozen:
            raise RuntimeError("Reference index is frozen.")
        if not long_unit_id.strip():
            raise ValueError("long_unit_id cannot be empty.")
        if len(dense_vector) == 0:
            raise ValueError("dense_vector cannot be empty.")
        vec = _l2_normalize(dense_vector)
        if self._dimension is None:
            self._dimension = len(vec)
        elif len(vec) != self._dimension:
            raise ValueError("Inconsistent dense-vector dimension.")
        self._ids.append(long_unit_id)
        self._vectors.append(vec)

    def freeze(self) -> None:
        self._frozen = True

    def __len__(self) -> int:
        return len(self._ids)

    @property
    def dimension(self) -> int | None:
        return self._dimension

    def search(
        self,
        query_vector: Sequence[float],
        limit: int,
    ) -> tuple[tuple[str, float, int], ...]:
        if limit <= 0:
            raise ValueError("limit must be positive.")
        if self._dimension is None:
            return ()
        if len(query_vector) != self._dimension:
            raise ValueError("Query-vector dimension mismatch.")
        q = _l2_normalize(query_vector)

        scored: list[tuple[str, float]] = []
        for long_unit_id, vec in zip(self._ids, self._vectors, strict=True):
            s = 0.0
            for a, b in zip(vec, q, strict=True):
                s += a * b
            scored.append((long_unit_id, s))

        scored.sort(key=lambda item: (-item[1], item[0]))
        return tuple(
            (long_unit_id, score, rank)
            for rank, (long_unit_id, score) in enumerate(
                scored[:limit], start=1
            )
        )