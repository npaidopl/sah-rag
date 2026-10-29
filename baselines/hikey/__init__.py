"""HiKEY reported baseline.

Paper: Shin et al., "HiKEY: Hierarchical Multimodal Retrieval for
Open-Domain Document Question Answering", arXiv:2605.29606v1.

Status: REPORTED. No official code exists; native execution is
impossible without reimplementation. Metrics below are transcribed
verbatim from the paper (Tables 1, 2) and must NOT be compared to
SAH-RAG runs on other datasets.
"""
from __future__ import annotations

__all__ = ["HiKEYReportedBaseline"]

from .adapter import HiKEYReportedBaseline  # noqa: E402