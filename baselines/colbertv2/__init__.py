"""ColBERTv2 baseline adapter for SAH-RAG v2 (native-only).

This package implements the ColBERTv2 late-interaction retrieval baseline
against the frozen `BaseBaseline` contract defined in `baselines/base.py`.

Public API:
    ColBERTv2Baseline

Architectural note:
    ColBERTv2 is a *retrieval-only* architecture. Its official paper
    (arXiv:2112.01488) and repository (stanford-futuredata/ColBERT) evaluate
    retrieval over MS MARCO / BEIR and do NOT include a generation stage.
    This adapter therefore runs in native mode only, and terminates at the
    retrieval output (`list[RetrievedChunk]`).

    Controlled mode is intentionally unsupported. ColBERTv2 requires a
    token-level BERT-based encoder for late interaction. Injecting BGE-M3
    (dense-only) would destroy the architecture rather than adapt it, so
    `supports_controlled = False`.

Lazy loading:
    The heavy dependencies (torch, colbert-ai, faiss) are imported lazily
    inside the adapter's methods. Importing this package is therefore safe
    in CPU-only / offline test environments.
"""

from __future__ import annotations

__all__ = ["ColBERTv2Baseline"]


def __getattr__(name: str):
    """PEP 562 lazy attribute: import the adapter only when accessed."""
    if name == "ColBERTv2Baseline":
        from .adapter import ColBERTv2Baseline

        return ColBERTv2Baseline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")