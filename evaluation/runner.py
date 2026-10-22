"""Evaluation runner for one (baseline, dataset, mode) cell."""
from __future__ import annotations

import importlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from baselines.base import BaseBaseline, BaselineConfig, BaselineStatus, RunMode
from datasets.unified import load_dataset
from evaluation.retrieval_metrics import mrr_at_k, ndcg_at_k, recall_at_k

CUTOFFS = [1, 3, 5, 10]


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _load_registry() -> dict:
    path = _project_root() / "configs" / "baselines_registry.yaml"
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _baseline_entry(registry: dict, name: str) -> dict:
    for b in registry["baselines"]:
        if b["name"] == name:
            return b
    raise KeyError(f"Baseline '{name}' not found in registry")


def _load_config_yaml(baseline_dir: Path, mode: str) -> dict:
    path = baseline_dir / f"config_{mode}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Missing baseline config: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _find_adapter_class(entry: dict, mode: str):
    name = entry["name"]
    candidates = [f"baselines.{name}", f"baselines.{name}_adapter"]
    module = None
    last_err: Exception | None = None
    for mod_name in candidates:
        try:
            module = importlib.import_module(mod_name)
            break
        except ImportError as exc:
            last_err = exc
    if module is None:
        raise ImportError(f"Cannot import adapter for '{name}' (tried {candidates}): {last_err}")
    for attr in dir(module):
        obj = getattr(module, attr)
        if isinstance(obj, type) and issubclass(obj, BaseBaseline) and obj is not BaseBaseline:
            return obj
    raise ImportError(f"No BaseBaseline subclass in module for '{name}'")


def _extract_query_and_gt(example: Any) -> tuple[str, list[str]]:
    query = getattr(example, "query", None)
    if query is not None:
        qtext = getattr(query, "text", None) or getattr(query, "query", None) or str(query)
    else:
        qtext = getattr(example, "query_text", None) or getattr(example, "text", "") or ""
    gt = getattr(example, "ground_truth", None)
    if gt is not None:
        gt_ids = getattr(gt, "doc_ids", None) or getattr(gt, "relevant_doc_ids", None) or getattr(gt, "ids", None) or []
    else:
        gt_ids = getattr(example, "gt_doc_ids", None) or getattr(example, "relevant_doc_ids", None) or []
    return str(qtext), list(gt_ids)


def run_cell(cell, output_root: Path) -> dict:
    project_root = _project_root()
    registry = _load_registry()
    entry = _baseline_entry(registry, cell.baseline)
    run_dir = output_root / cell.name
    run_dir.mkdir(parents=True, exist_ok=True)

    if entry.get("status") == BaselineStatus.REPORTED.value:
        return _run_reported(cell, entry, run_dir, project_root)

    if cell.mode != "native":
        raise NotImplementedError(f"v2 supports native mode only for now; got {cell.mode!r}")

    baseline_dir = project_root / entry["path"]
    if not baseline_dir.exists():
        raise FileNotFoundError(f"Baseline directory missing: {baseline_dir}")

    cfg_yaml = _load_config_yaml(baseline_dir, cell.mode)
    cfg = BaselineConfig(name=cell.baseline, mode=RunMode(cell.mode), top_k=cell.top_k, extra=dict(cfg_yaml.get("extra") or {}))
    work_dir = run_dir / "work"
    work_dir.mkdir(parents=True, exist_ok=True)
    adapter_cls = _find_adapter_class(entry, cell.mode)
    baseline = adapter_cls(config=cfg, work_dir=work_dir, embedder=None, llm=None)

    ds = load_dataset(cell.dataset)
    started = datetime.now(timezone.utc).isoformat()
    baseline.build_index(ds.documents)

    buckets: dict[str, list[float]] = {}
    for k in CUTOFFS:
        buckets[f"recall@{k}"] = []
        buckets[f"mrr@{k}"] = []
        buckets[f"ndcg@{k}"] = []

    predictions_path = run_dir / "predictions.jsonl"
    n_evaluated = 0
    with predictions_path.open("w", encoding="utf-8") as pf:
        for example in ds.examples:
            qtext, gt_ids = _extract_query_and_gt(example)
            if not qtext:
                continue
            results = baseline.retrieve(qtext, top_k=cell.top_k)
            retrieved_ids = [r.doc_id for r in results]
            for k in CUTOFFS:
                buckets[f"recall@{k}"].append(recall_at_k(retrieved_ids, gt_ids, k))
                buckets[f"mrr@{k}"].append(mrr_at_k(retrieved_ids, gt_ids, k))
                buckets[f"ndcg@{k}"].append(ndcg_at_k(retrieved_ids, gt_ids, k))
            record = {
                "query": qtext,
                "gt_doc_ids": gt_ids,
                "retrieved": [{"doc_id": r.doc_id, "chunk_id": r.chunk_id, "rank": r.rank, "score": r.score} for r in results],
            }
            pf.write(json.dumps(record) + "\n")
            n_evaluated += 1

    aggregated = {name: (sum(v) / len(v) if v else 0.0) for name, v in buckets.items()}
    metrics = {
        "run_id": cell.run_id, "baseline": cell.baseline, "dataset": cell.dataset, "mode": cell.mode,
        "top_k": cell.top_k, "n_documents": len(ds.documents), "n_examples": len(ds.examples),
        "n_evaluated": n_evaluated, "started_at": started,
        "finished_at": datetime.now(timezone.utc).isoformat(), "metrics": aggregated,
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def _run_reported(cell, entry: dict, run_dir: Path, project_root: Path) -> dict:
    reported_path = project_root / entry["path"] / "reported_metrics.yaml"
    if not reported_path.exists():
        raise FileNotFoundError(f"reported_metrics.yaml missing for {cell.baseline}: {reported_path}")
    with reported_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    out = {"run_id": cell.run_id, "baseline": cell.baseline, "dataset": cell.dataset, "mode": cell.mode, "reported": True, "metrics": data.get(cell.dataset, {})}
    (run_dir / "metrics.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


__all__ = ["run_cell", "CUTOFFS"]