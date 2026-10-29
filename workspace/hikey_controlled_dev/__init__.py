"""HiKEY controlled baseline package (SAH-RAG v2).

Paper-derived controlled comparator for:
    Shin et al., "HiKEY: Hierarchical Multimodal Retrieval for
    Open-Domain Document Question Answering", ACL 2026.
    arXiv:2605.29606

Native track: NOT supported (no official code exists).
Controlled track: paper-derived text-only approximation.
"""
from __future__ import annotations

__all__ = ["HiKEYBaseline"]

from .adapter import HiKEYBaseline  # noqa: E402