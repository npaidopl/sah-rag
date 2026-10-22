"""Manifest parser for the SAH-RAG orchestrator (v2)."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass(frozen=True)
class RunCell:
    baseline: str
    dataset: str
    mode: str
    top_k: int = 10
    extra: dict = field(default_factory=dict)

    @property
    def run_id(self) -> str:
        payload = f"{self.baseline}|{self.dataset}|{self.mode}|{self.top_k}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    @property
    def name(self) -> str:
        return f"{self.mode}__{self.baseline}__{self.dataset}__{self.run_id}"


@dataclass
class RunManifest:
    modes: list[str]
    baselines: list[str]
    datasets: list[str]
    top_k: int = 10
    output_root: Path | None = None
    extra: dict = field(default_factory=dict)

    def cells(self) -> list[RunCell]:
        out: list[RunCell] = []
        for mode in self.modes:
            for baseline in self.baselines:
                for dataset in self.datasets:
                    out.append(RunCell(baseline=baseline, dataset=dataset, mode=mode, top_k=self.top_k))
        return out


def load_manifest(path: Path) -> RunManifest:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    output_root = data.get("output_root")
    if output_root:
        output_root = Path(output_root)
    return RunManifest(
        modes=list(data["modes"]),
        baselines=list(data["baselines"]),
        datasets=list(data["datasets"]),
        top_k=int(data.get("top_k", 10)),
        output_root=output_root,
        extra=dict(data.get("extra", {})),
    )