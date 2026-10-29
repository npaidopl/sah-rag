"""HiKEY native pipeline (paper-exact reimplementation).

Paper: Shin et al., "HiKEY: Hierarchical Multimodal Retrieval for
Open-Domain Document Question Answering", arXiv:2605.29606v1.

This is a STANDALONE pipeline (not a v2 BaseBaseline adapter) because
HiKEY consumes PDFs and multimodal evidence, not the flat-text
CanonicalDocument format.

Native mode entry: python run.py --corpus <pdf_dir> --queries <json>

The retrieval pipeline is fully IR-compatible (v2 is IR-only). The
LVLM reader is included as a separate component for the optional
end-to-end QA path.
"""
from __future__ import annotations

__all__ = ["HiKEYNativePipeline"]

from .pipeline import HiKEYNativePipeline  # noqa: E402