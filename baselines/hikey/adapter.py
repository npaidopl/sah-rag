"""Reported-only HiKEY adapter."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from baselines.base import (
    BaselineConfig,
    BaselineStatus,
    ReportedBaseline,
)


class HiKEYReportedBaseline(ReportedBaseline):
    name = "hikey"
    version = "0.2.0"
    status = BaselineStatus.REPORTED
    supports_native = False
    supports_controlled = False
    is_llm_based = False

    def __init__(
        self,
        *,
        config: BaselineConfig,
        work_dir: Path,
        embedder=None,
        llm=None,
    ) -> None:
        super().__init__(config=config, work_dir=work_dir, embedder=embedder, llm=llm)
        self._metrics: dict[str, dict[str, float]] = {}
        metrics_path = Path(__file__).parent / "reported_metrics.yaml"
        if metrics_path.exists():
            data = yaml.safe_load(metrics_path.read_text(encoding="utf-8"))
            self._metrics = data.get("datasets", {})

    def get_reported_metrics(self, dataset_id: str) -> dict[str, float]:
        return dict(self._metrics.get(dataset_id, {}))