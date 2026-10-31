"""PathRAG reported baseline (paper numbers only).

Classification: REPORTED.

Rationale (full detail in README.md):
  1. Official repo has no LICENSE file - cannot redistribute.
  2. Wrapper with available local LLMs produced 0 edges -> not PathRAG.
  3. Controlled mode requires injected LLM; only mock was validated.

The working wrapper is retained at workspace/pathrag_wrapper_future/
for a future GPU/API-key holder. It is NOT a graduation candidate.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from baselines.base import (
    BaselineStatus,
    ReportedBaseline,
)


class PathRAGReportedBaseline(ReportedBaseline):
    """Reported-only PathRAG baseline.

    Metrics sourced from the PathRAG paper (arXiv 2502.14902, AAAI-26).
    See reported_metrics.yaml for the data structure.
    """

    name = "pathrag"
    version = "1.0.0"
    status = BaselineStatus.REPORTED
    supports_native = False
    supports_controlled = False
    is_llm_based = True

    def __init__(self, *, config, work_dir, embedder=None, llm=None):
        super().__init__(config=config, work_dir=work_dir, embedder=embedder, llm=llm)
        self._metrics_cache: dict[str, Any] | None = None

    def _load_metrics(self) -> dict[str, Any]:
        if self._metrics_cache is not None:
            return self._metrics_cache
        path = Path(__file__).parent / "reported_metrics.yaml"
        if not path.exists():
            self._metrics_cache = {}
            return self._metrics_cache
        with path.open(encoding="utf-8") as f:
            self._metrics_cache = yaml.safe_load(f) or {}
        return self._metrics_cache

    def get_reported_metrics(self, dataset_id: str) -> dict[str, float]:
        data = self._load_metrics()
        datasets = data.get("datasets") or {}
        entry = datasets.get(dataset_id) or {}
        return {k: float(v) for k, v in entry.items() if isinstance(v, (int, float))}
