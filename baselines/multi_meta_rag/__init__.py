"""Multi-Meta-RAG baseline adapter for SAH-RAG v2.

Public API:
    from baselines.multi_meta_rag import MultiMetaRAGBaseline

The relative import below works both in workspace/ and in baselines/,
following the pattern used by baselines/hipporag2/__init__.py.
"""
from __future__ import annotations

from .adapter import MultiMetaRAGBaseline

__all__ = ["MultiMetaRAGBaseline"]