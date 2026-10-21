"""Dense-only BGE-M3 encoder bound to a pinned local snapshot.

FlagEmbedding is imported lazily so that importing this module does not
require the embedding environment. The encoder never reaches the network:
callers must point it at an already-materialized snapshot directory.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

BGE_M3_MODEL_ID = "BAAI/bge-m3"
BGE_M3_MODEL_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
BGE_M3_DENSE_DIMENSION = 1024
BGE_M3_SPECIAL_TOKEN_COUNT = 2


class BgeM3DenseEncoder:
    """Dense-only encoder using FlagEmbedding.BGEM3FlagModel."""

    def __init__(
        self,
        *,
        snapshot_path: Path | str,
        device: str = "cpu",
        use_fp16: bool = False,
    ) -> None:
        snapshot = Path(snapshot_path)
        if not snapshot.exists():
            raise FileNotFoundError(f"BGE-M3 snapshot not found: {snapshot}")
        from FlagEmbedding import (  # type: ignore[import-untyped]
            BGEM3FlagModel,
        )

        self._snapshot_path = snapshot
        self._model = BGEM3FlagModel(
            str(snapshot),
            use_fp16=use_fp16,
            device=device,
        )

    @property
    def snapshot_path(self) -> Path:
        return self._snapshot_path

    def encode_dense(
        self,
        texts: Sequence[str],
        *,
        batch_size: int,
        max_length: int,
    ) -> tuple[tuple[float, ...], ...]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if max_length <= 0:
            raise ValueError("max_length must be positive.")
        if len(texts) == 0:
            return ()

        output = self._model.encode(
            list(texts),
            batch_size=batch_size,
            max_length=max_length,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        dense = output["dense_vecs"]
        rows: list[tuple[float, ...]] = []
        for row in dense:
            rows.append(tuple(float(x) for x in row))
        return tuple(rows)