"""HippoRAG2 baseline adapter for SAH-RAG v2.

Public API:
    from baselines.hipporag2 import HippoRAG2Baseline

The relative import below works both in workspace/hipporag2_dev/ and
after graduation in baselines/hipporag2/ - no change needed on move.
"""
from __future__ import annotations

from .adapter import HippoRAG2Baseline

__all__ = ["HippoRAG2Baseline"]