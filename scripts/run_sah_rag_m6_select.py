# ruff: noqa: E402
"""SAH-RAG M6 - Best-config selection via one-standard-error rule + Pareto.

Inputs:
  - _m5_confirmatory_results.jsonl (per config, per dataset: metrics, elapsed)
  - _m5_per_query_metrics.jsonl (per query, per config: metrics)
  - _m2_grid_root_manifest.json (chunk counts for storage proxy)

Outputs:
  - outputs/external_sah_rag/_m6_selection.json (decision record)
  - baseline_audit/my_rag_m6_frozen_config.yaml (frozen config)
  - outputs/external_sah_rag/_m6_pareto.json (Pareto frontier)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np
import yaml

OUT_ROOT = REPO / "outputs" / "external_sah_rag"
BASELINE = REPO / "baseline_audit"

DATASETS = ("clerk", "cuad", "greek_bar_bench", "lara")
PRIMARY_METRIC = "recall@10"
SECONDARY_METRICS = ("hit@10", "ndcg@10", "mrr@10", "precision@10")
BOOTSTRAP_B = 1000
BOOTSTRAP_SEED = 42
FP16_BYTES_PER_VALUE = 2
DENSE_DIMENSION = 1024


# -----------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------
def parse_config_id(cid: str) -> dict[str, Any]:
    """c256_o128_hybrid_rrf_k200 -> {cs:256, ov:128, mode:hybrid_rrf, K:200}."""
    m = re.match(r"c(\d+)_o(\d+)_(.+)_k(\d+)", cid)
    if not m:
        raise ValueError(f"bad config_id: {cid}")
    return {
        "chunk_size": int(m.group(1)),
        "chunk_overlap": int(m.group(2)),
        "retrieval_mode": m.group(3),
        "candidate_limit": int(m.group(4)),
    }


def load_confirmatory() -> dict[str, dict[str, dict[str, Any]]]:
    """{config_id: {dataset: row}}."""
    p = OUT_ROOT / "_m5_confirmatory_results.jsonl"
    out: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            out[r["config_id"]][r["dataset"]] = r
    return dict(out)


def load_per_query() -> dict[str, dict[str, dict[str, float]]]:
    """{config_id: {dataset: {query_id: recall@10}}}."""
    p = OUT_ROOT / "_m5_per_query_metrics.jsonl"
    out: dict[str, dict[str, dict[str, float]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            out[r["config_id"]][r["dataset"]][r["query_id"]] = float(
                r[PRIMARY_METRIC]
            )
    return {c: dict(d) for c, d in out.items()}


def load_chunk_counts() -> dict[tuple[str, int, int], int]:
    """{(dataset, chunk_size, overlap): chunk_count}."""
    p = OUT_ROOT / "_m2_grid_root_manifest.json"
    j = json.loads(p.read_text(encoding="utf-8"))
    out: dict[tuple[str, int, int], int] = {}
    for v in j["variants"]:
        out[(v["dataset_id"], v["chunk_size"], v["chunk_overlap"])] = int(
            v["chunk_count"]
        )
    return out


# -----------------------------------------------------------------
# Bootstrap
# -----------------------------------------------------------------
def paired_bootstrap_macro(
    pq: dict[str, dict[str, dict[str, float]]],
    configs: list[str],
    datasets: tuple[str, ...],
    *,
    b: int = BOOTSTRAP_B,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, np.ndarray]:
    """Return {config_id: (B,) array of bootstrap macro R@10}.

    Paired by resampling queries once per iteration (shared across configs).
    """
    rng = np.random.default_rng(seed)

    # Pre-extract query ids per dataset (assume same across configs)
    ref_config = configs[0]
    qids_per_ds: dict[str, list[str]] = {
        ds: sorted(pq[ref_config][ds].keys()) for ds in datasets
    }

    # Pre-extract per-config R@10 arrays per dataset, aligned to qids
    arrs: dict[str, dict[str, np.ndarray]] = defaultdict(dict)
    for c in configs:
        for ds in datasets:
            qids = qids_per_ds[ds]
            arrs[c][ds] = np.array(
                [pq[c][ds][q] for q in qids], dtype=np.float32
            )

    results: dict[str, np.ndarray] = {
        c: np.zeros(b, dtype=np.float64) for c in configs
    }

    for it in range(b):
        ds_means_per_c: dict[str, list[float]] = {c: [] for c in configs}
        for ds in datasets:
            n = len(qids_per_ds[ds])
            idx = rng.integers(0, n, size=n)
            for c in configs:
                ds_means_per_c[c].append(
                    float(arrs[c][ds][idx].mean())
                )
        for c in configs:
            results[c][it] = float(np.mean(ds_means_per_c[c]))

    return results


def one_se_rule(
    bootstrap: dict[str, np.ndarray],
    configs: list[str],
) -> dict[str, Any]:
    """Return best config and set of configs within 1 SE (paired)."""
    means = {c: float(bootstrap[c].mean()) for c in configs}
    best = max(configs, key=lambda c: means[c])
    best_dist = bootstrap[best]

    retained = [best]
    details: dict[str, dict[str, float]] = {
        best: {
            "mean": means[best],
            "mean_delta_vs_best": 0.0,
            "se_delta_vs_best": 0.0,
            "within_1se": True,
        }
    }

    for c in configs:
        if c == best:
            continue
        deltas = best_dist - bootstrap[c]  # paired
        mean_delta = float(deltas.mean())
        se_delta = float(deltas.std(ddof=1)) if deltas.size > 1 else 0.0
        within = mean_delta <= se_delta
        details[c] = {
            "mean": means[c],
            "mean_delta_vs_best": mean_delta,
            "se_delta_vs_best": se_delta,
            "within_1se": within,
        }
        if within:
            retained.append(c)

    return {
        "best_config_id": best,
        "best_mean_r10": means[best],
        "se_of_best": float(best_dist.std(ddof=1)),
        "retained_configs": sorted(retained),
        "per_config_details": details,
    }


# -----------------------------------------------------------------
# Cost proxies
# -----------------------------------------------------------------
def compute_latency(
    confirm: dict[str, dict[str, dict[str, Any]]],
    configs: list[str],
    datasets: tuple[str, ...],
) -> dict[str, float]:
    """Mean over datasets of (elapsed_seconds / query_count)."""
    out: dict[str, float] = {}
    for c in configs:
        per_ds: list[float] = []
        for ds in datasets:
            row = confirm[c].get(ds)
            if row is None:
                continue
            per_ds.append(row["elapsed_seconds"] / row["query_count"])
        out[c] = float(np.mean(per_ds)) if per_ds else float("inf")
    return out


def compute_storage(
    chunk_counts: dict[tuple[str, int, int], int],
    configs: list[str],
    datasets: tuple[str, ...],
) -> dict[str, float]:
    """Storage proxy: total points across datasets x bytes/point (dense FP16)."""
    out: dict[str, float] = {}
    for c in configs:
        meta = parse_config_id(c)
        cs = meta["chunk_size"]
        ov = meta["chunk_overlap"]
        total_bytes = 0
        for ds in datasets:
            n = chunk_counts.get((ds, cs, ov), 0)
            total_bytes += n * DENSE_DIMENSION * FP16_BYTES_PER_VALUE
        out[c] = float(total_bytes)
    return out


# -----------------------------------------------------------------
# Selection
# -----------------------------------------------------------------
def lexicographic_tiebreak(
    retained: list[str],
    latency: dict[str, float],
    storage: dict[str, float],
) -> list[str]:
    """Sort by: lower latency, lower storage, smaller chunk_size,
    lower candidate_k, config_id ascending."""

    def key(c: str) -> tuple:
        meta = parse_config_id(c)
        return (
            latency[c],
            storage[c],
            meta["chunk_size"],
            meta["candidate_limit"],
            c,
        )

    return sorted(retained, key=key)


def pareto_frontier(
    configs: list[str],
    primary: dict[str, float],
    latency: dict[str, float],
    storage: dict[str, float],
) -> list[str]:
    """Non-dominated set: maximize primary, minimize latency, minimize storage."""
    frontier: list[str] = []
    for a in configs:
        dominated = False
        for b in configs:
            if a == b:
                continue
            if (
                primary[b] >= primary[a]
                and latency[b] <= latency[a]
                and storage[b] <= storage[a]
                and (
                    primary[b] > primary[a]
                    or latency[b] < latency[a]
                    or storage[b] < storage[a]
                )
            ):
                dominated = True
                break
        if not dominated:
            frontier.append(a)
    return sorted(frontier)


# -----------------------------------------------------------------
# Writers
# -----------------------------------------------------------------
def write_json_atomic(p: Path, d: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(
        json.dumps(d, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tmp.replace(p)


def write_yaml_atomic(p: Path, d: dict[str, Any]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(
        yaml.safe_dump(d, sort_keys=True, allow_unicode=True),
        encoding="utf-8",
    )
    tmp.replace(p)


def build_frozen_config(
    selected: str,
    selection: dict[str, Any],
    latency: dict[str, float],
    storage: dict[str, float],
    confirm: dict[str, dict[str, dict[str, Any]]],
    chunk_counts: dict[tuple[str, int, int], int],
) -> dict[str, Any]:
    meta = parse_config_id(selected)
    per_ds_metrics = {
        ds: {
            "macro": confirm[selected][ds]["macro"],
            "query_count": confirm[selected][ds]["query_count"],
        }
        for ds in DATASETS if ds in confirm[selected]
    }
    return {
        "artifact_type": "sah_rag_m6_frozen_config",
        "artifact_version": "1.0",
        "selection_rule": "one_standard_error_then_lexicographic",
        "selection_rule_details": {
            "primary_metric": PRIMARY_METRIC,
            "bootstrap_iterations": BOOTSTRAP_B,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "tie_break_priority": [
                "lower_latency_per_query",
                "lower_storage_bytes",
                "smaller_chunk_size",
                "lower_candidate_limit",
                "config_id_ascending",
            ],
        },
        "selected_config_id": selected,
        "selected_config": {
            "chunk_size": meta["chunk_size"],
            "chunk_overlap": meta["chunk_overlap"],
            "retrieval_mode": meta["retrieval_mode"],
            "candidate_limit": meta["candidate_limit"],
        },
        "selected_config_metrics": {
            "macro_recall@10": selection["best_mean_r10"]
            if selected == selection["best_config_id"]
            else selection["per_config_details"][selected]["mean"],
            "macro_recall@10_se": selection["se_of_best"]
            if selected == selection["best_config_id"]
            else selection["per_config_details"][selected]["se_delta_vs_best"],
            "per_dataset": per_ds_metrics,
        },
        "cost_proxies": {
            "mean_latency_per_query_seconds": latency[selected],
            "total_dense_storage_bytes_proxy": storage[selected],
        },
        "retained_within_1se": selection["retained_configs"],
        "best_by_mean_r10": selection["best_config_id"],
        "chunk_counts_per_dataset": {
            ds: chunk_counts.get(
                (ds, meta["chunk_size"], meta["chunk_overlap"]), 0,
            )
            for ds in DATASETS
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.parse_args()

    print("[M6] loading M5 confirmatory results ...")
    confirm = load_confirmatory()
    configs = sorted(confirm.keys())
    print(f"[M6] {len(configs)} configs: {configs}")

    print("[M6] loading per-query metrics ...")
    pq = load_per_query()

    print("[M6] loading chunk counts for storage proxy ...")
    chunk_counts = load_chunk_counts()

    print(f"[M6] bootstrap (B={BOOTSTRAP_B}, seed={BOOTSTRAP_SEED}) ...")
    bootstrap = paired_bootstrap_macro(pq, configs, DATASETS)

    print("[M6] applying one-standard-error rule ...")
    selection = one_se_rule(bootstrap, configs)

    print("[M6] computing cost proxies ...")
    latency = compute_latency(confirm, configs, DATASETS)
    storage = compute_storage(chunk_counts, configs, DATASETS)

    print("[M6] lexicographic tie-break within 1-SE band ...")
    ordered = lexicographic_tiebreak(
        selection["retained_configs"], latency, storage,
    )
    selected = ordered[0]

    print("[M6] Pareto frontier ...")
    primary_means = {
        c: float(bootstrap[c].mean()) for c in configs
    }
    pareto = pareto_frontier(configs, primary_means, latency, storage)

    # write outputs
    sel_path = OUT_ROOT / "_m6_selection.json"
    write_json_atomic(sel_path, {
        "artifact_type": "sah_rag_m6_selection",
        "artifact_version": "1.0",
        "execution_id": "sah-rag-m6-v1",
        "primary_metric": PRIMARY_METRIC,
        "bootstrap": {
            "iterations": BOOTSTRAP_B,
            "seed": BOOTSTRAP_SEED,
        },
        "selection": selection,
        "ordered_within_1se": ordered,
        "latency_seconds_per_query": latency,
        "storage_bytes_proxy": storage,
        "primary_means": primary_means,
        "pareto_frontier": pareto,
        "selected_config_id": selected,
    })

    frozen = build_frozen_config(
        selected, selection, latency, storage, confirm, chunk_counts,
    )
    write_yaml_atomic(BASELINE / "sah_rag_m6_frozen_config.yaml", frozen)

    write_json_atomic(OUT_ROOT / "_m6_pareto.json", {
        "artifact_type": "sah_rag_m6_pareto",
        "artifact_version": "1.0",
        "frontier": pareto,
        "configs": {
            c: {
                "macro_r10": primary_means[c],
                "latency_seconds_per_query": latency[c],
                "storage_bytes_proxy": storage[c],
            }
            for c in configs
        },
    })

    # console report
    print()
    print("=" * 90)
    print("M6 BEST-CONFIG SELECTION (one-standard-error rule)")
    print("=" * 90)
    print(f"Best by mean R@10: {selection['best_config_id']} "
          f"(R@10={selection['best_mean_r10']:.4f}, "
          f"SE={selection['se_of_best']:.4f})")
    print()
    print("Configs within 1 SE (paired bootstrap):")
    for c in selection["retained_configs"]:
        d = selection["per_config_details"][c]
        print(
            f"  {c:40s}  R@10={d['mean']:.4f}  "
            f"mean_delta={d['mean_delta_vs_best']:+.4f}  "
            f"SE_delta={d['se_delta_vs_best']:.4f}"
        )
    print()
    print("After lexicographic tie-break:")
    print("  1. lower latency  -> 2. lower storage -> "
          "3. smaller chunk_size -> 4. lower candidate_k -> "
          "5. config_id asc")
    print("  Selection order:")
    for i, c in enumerate(ordered, 1):
        print(
            f"    {i}. {c:40s}  "
            f"lat={latency[c]:.4f}s/q  "
            f"storage={storage[c]/1e6:.1f} MB"
        )
    print()
    print(f"SELECTED: {selected}")
    print()
    print("Pareto frontier (R@10 max, latency min, storage min):")
    for c in pareto:
        print(
            f"  {c:40s}  R@10={primary_means[c]:.4f}  "
            f"lat={latency[c]:.4f}s/q  "
            f"storage={storage[c]/1e6:.1f} MB"
        )
    print()
    print(f"M6 COMPLETE β€” frozen config at "
          f"baseline_audit/my_rag_m6_frozen_config.yaml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())