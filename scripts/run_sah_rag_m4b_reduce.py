# ruff: noqa: E402
"""SAH-RAG M4B - Reduce the screening results.

Reads _m4a_screening_results.jsonl and produces:
  - per-axis response curves (chunk_size, overlap, mode, candidate_k)
  - decision-rule verdicts per hypothesis (monotonicity, saturation)
  - dominance table
  - top-2 recommended values per axis
  - reduced factorial candidate list (max 48 configs)
  - baseline comparison

Writes:
  - outputs/external_sah_rag/_m4b_reduction.json
  - outputs/external_sah_rag/_m4b_response_curves.json
  - outputs/external_sah_rag/_m4b_reduced_candidates.jsonl
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

OUT_ROOT = REPO / "outputs" / "external_sah_rag"
M4A_JSONL = OUT_ROOT / "_m4a_screening_results.jsonl"

DATASETS = ("clerk", "cuad", "greek_bar_bench", "lara")
PRIMARY_METRIC = "recall@10"
SECONDARY_METRICS = ("hit@10", "ndcg@10", "mrr@10", "precision@10")
SATURATION_THRESHOLD = 0.01
MONOTONICITY_DIRECTION = 0.80


def load_results() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with M4A_JSONL.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def aggregate_per_config(
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, float]]:
    """Aggregate per-config across datasets: mean of primary + secondaries."""
    by_cfg: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_cfg[r["config_id"]].append(r)

    out: dict[str, dict[str, float]] = {}
    for cid, cfg_rows in by_cfg.items():
        if len(cfg_rows) != len(DATASETS):
            continue
        metrics: dict[str, float] = {}
        for m in (PRIMARY_METRIC, *SECONDARY_METRICS):
            metrics[m] = float(statistics.mean(
                r["macro"][m] for r in cfg_rows
            ))
        out[cid] = metrics
    return out


def build_response_curves(
    rows: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """One-factor response curves: hold 3 of 4 axes at default."""
    DEFAULT = {
        "chunk_size": 256,
        "chunk_overlap": 32,
        "retrieval_mode": "hybrid_rrf",
        "candidate_limit": 200,
    }
    curves: dict[str, list[dict[str, Any]]] = {
        "chunk_size": [],
        "chunk_overlap": [],
        "retrieval_mode": [],
        "candidate_limit": [],
    }

    for axis in curves:
        values = {
            r[axis] for r in rows
            if all(
                r[other] == DEFAULT[other]
                for other in DEFAULT
                if other != axis
            )
        }
        # numeric axes sort numerically; string axes sort alphabetically
        if all(isinstance(v, (int, float)) for v in values):
            sorted_values = sorted(values)
        else:
            sorted_values = sorted(values, key=lambda v: str(v))

        for value in sorted_values:
            selected = [
                r for r in rows
                if r[axis] == value
                and all(
                    r[o] == DEFAULT[o] for o in DEFAULT if o != axis
                )
            ]
            if not selected:
                continue
            primary = float(statistics.mean(
                r["macro"][PRIMARY_METRIC] for r in selected
            ))
            secondaries = {
                m: float(statistics.mean(
                    r["macro"][m] for r in selected
                )) for m in SECONDARY_METRICS
            }
            curves[axis].append({
                "value": value,
                "dataset_count": len(selected),
                PRIMARY_METRIC: primary,
                **secondaries,
            })
    return curves


def check_monotonicity(
    curve: list[dict[str, Any]],
) -> dict[str, Any]:
    """β‰¥80% adjacent steps same direction + nonzero total change."""
    if len(curve) < 2:
        return {"verdict": "INSUFFICIENT_DATA"}
    values = [c[PRIMARY_METRIC] for c in curve]
    deltas = [values[i + 1] - values[i] for i in range(len(values) - 1)]
    nonzero = [d for d in deltas if abs(d) > 1e-9]
    if not nonzero:
        return {"verdict": "NO_CHANGE"}
    positive = sum(1 for d in nonzero if d > 0)
    negative = sum(1 for d in nonzero if d < 0)
    dominant = max(positive, negative) / len(nonzero)
    total_change = values[-1] - values[0]
    direction = "increasing" if positive >= negative else "decreasing"
    return {
        "verdict": (
            "MONOTONE"
            if dominant >= MONOTONICITY_DIRECTION
            and abs(total_change) > 1e-6
            else "NON_MONOTONE"
        ),
        "direction": direction,
        "dominant_fraction": round(dominant, 3),
        "total_change": round(total_change, 6),
    }


def find_saturation(
    curve: list[dict[str, Any]],
    threshold: float = SATURATION_THRESHOLD,
) -> dict[str, Any]:
    """Smallest value v such that all subsequent marginal gains < threshold."""
    if len(curve) < 2:
        return {"verdict": "INSUFFICIENT_DATA"}
    values = [c[PRIMARY_METRIC] for c in curve]
    for i in range(len(values) - 1):
        remaining = [
            values[j + 1] - values[j]
            for j in range(i, len(values) - 1)
        ]
        if all(abs(d) < threshold for d in remaining):
            return {
                "verdict": "SATURATED",
                "saturation_value": curve[i]["value"],
                "marginal_gains": [
                    round(d, 6) for d in remaining
                ],
            }
    return {"verdict": "NOT_SATURATED_IN_GRID"}


def dominance_analysis(
    agg: dict[str, dict[str, float]],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Dominance: A dominates B if quality β‰¥ and cost β‰¤, at least one strict."""
    # Cost proxy: chunk count (from results.jsonl we have chunk_size/overlap,
    # but not storage; approximate cost as inverse of chunk_size * inv overlap)
    # Instead use a composite cost: 1 - chunk_size/1024 (smaller chunks = cost)
    # We'll use chunk_count as storage proxy since we have it per row.
    by_cfg_chunks: dict[str, float] = {}
    for r in rows:
        cid = r["config_id"]
        # approximate chunk count using chunk_size/overlap + data point
        # we stored query_count only, use chunk_size as cost
        cs = r["chunk_size"]
        ov = r["chunk_overlap"]
        by_cfg_chunks[cid] = 1.0 / (cs + ov)  # smaller -> bigger cost

    dominated: list[str] = []
    dominators: dict[str, list[str]] = defaultdict(list)
    for a, ma in agg.items():
        for b, mb in agg.items():
            if a == b:
                continue
            quality_a = ma[PRIMARY_METRIC]
            quality_b = mb[PRIMARY_METRIC]
            cost_a = by_cfg_chunks.get(a, 1.0)
            cost_b = by_cfg_chunks.get(b, 1.0)
            if (
                quality_a >= quality_b
                and cost_a <= cost_b
                and (quality_a > quality_b or cost_a < cost_b)
            ):
                dominators[b].append(a)

    for cid in agg:
        if dominators.get(cid):
            dominated.append(cid)

    return {
        "dominated_count": len(dominated),
        "dominated_configs": sorted(dominated),
        "non_dominated_count": len(agg) - len(dominated),
    }


def recommend_top2(curves: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Pick top-2 values per axis by primary metric, dedupe."""
    rec: dict[str, list[Any]] = {}
    for axis, curve in curves.items():
        sorted_curve = sorted(
            curve, key=lambda c: -c[PRIMARY_METRIC],
        )
        top = []
        for c in sorted_curve:
            if len(top) >= 2:
                break
            top.append({
                "value": c["value"],
                PRIMARY_METRIC: c[PRIMARY_METRIC],
            })
        rec[axis] = top
    return rec


def build_reduced_factorial(
    recommendations: dict[str, Any],
) -> list[dict[str, Any]]:
    """Cross product of top-2 per axis, capped at 48."""
    sizes = [r["value"] for r in recommendations["chunk_size"]]
    overlaps = [r["value"] for r in recommendations["chunk_overlap"]]
    modes = [r["value"] for r in recommendations["retrieval_mode"]]
    ks = [r["value"] for r in recommendations["candidate_limit"]]

    combos: list[dict[str, Any]] = []
    for cs in sizes:
        for ov in overlaps:
            for mode in modes:
                for k in ks:
                    combos.append({
                        "chunk_size": cs,
                        "chunk_overlap": ov,
                        "retrieval_mode": mode,
                        "candidate_limit": k,
                        "config_id": f"c{cs}_o{ov}_{mode}_k{k}",
                    })
    # cap
    combos = combos[:48]
    # ensure default is included
    default = {
        "chunk_size": 256,
        "chunk_overlap": 32,
        "retrieval_mode": "hybrid_rrf",
        "candidate_limit": 200,
        "config_id": "c256_o32_hybrid_rrf_k200",
    }
    if default["config_id"] not in {c["config_id"] for c in combos}:
        combos.insert(0, default)
    return combos[:48]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true")
    ap.parse_args()

    print(f"[M4B] loading {M4A_JSONL} ...")
    rows = load_results()
    print(f"[M4B] {len(rows)} rows")

    agg = aggregate_per_config(rows)
    print(f"[M4B] {len(agg)} unique configs")

    print("[M4B] building response curves ...")
    curves = build_response_curves(rows)

    print("[M4B] applying decision rules ...")
    verdicts: dict[str, Any] = {}
    for axis, curve in curves.items():
        verdicts[axis] = {
            "monotonicity": check_monotonicity(curve),
            "saturation": find_saturation(curve),
        }

    print("[M4B] dominance analysis ...")
    dominance = dominance_analysis(agg, rows)

    print("[M4B] top-2 recommendations ...")
    recommendations = recommend_top2(curves)

    print("[M4B] reduced factorial ...")
    combos = build_reduced_factorial(recommendations)

    # Write outputs
    curves_path = OUT_ROOT / "_m4b_response_curves.json"
    curves_path.write_text(
        json.dumps(curves, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    reduction = {
        "artifact_type": "sah_rag_m4b_reduction",
        "artifact_version": "1.0",
        "execution_id": "sah-rag-m4b-v1",
        "row_count": len(rows),
        "unique_configs": len(agg),
        "response_curves_ref": str(curves_path.name),
        "verdicts": verdicts,
        "dominance": dominance,
        "top2_recommendations": recommendations,
        "reduced_factorial_size": len(combos),
        "primary_metric": PRIMARY_METRIC,
    }
    reduction_path = OUT_ROOT / "_m4b_reduction.json"
    reduction_path.write_text(
        json.dumps(reduction, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    cand_path = OUT_ROOT / "_m4b_reduced_candidates.jsonl"
    with cand_path.open("w", encoding="utf-8", newline="\n") as fh:
        for c in combos:
            fh.write(json.dumps(c, ensure_ascii=False, sort_keys=True,
                                separators=(",", ":")))
            fh.write("\n")

    # console summary
    print()
    print("=" * 80)
    print("RESPONSE CURVES (primary = macro R@10)")
    print("=" * 80)
    for axis, curve in curves.items():
        print(f"\n[{axis}]")
        for c in curve:
            print(
                f"  value={str(c['value']):>6s} "
                f"R@10={c[PRIMARY_METRIC]:.4f} "
                f"H@10={c['hit@10']:.4f} "
                f"nDCG@10={c['ndcg@10']:.4f}"
            )
        v = verdicts[axis]
        print(f"  β†’ monotonicity: {v['monotonicity']}")
        print(f"  β†’ saturation:   {v['saturation']}")

    print()
    print("=" * 80)
    print("TOP-2 PER AXIS")
    print("=" * 80)
    for axis, recs in recommendations.items():
        values = [r["value"] for r in recs]
        print(f"  {axis:20s} -> {values}")

    print()
    print("=" * 80)
    print("DOMINANCE")
    print("=" * 80)
    print(f"  dominated configs:  {dominance['dominated_count']}")
    print(f"  non-dominated:      {dominance['non_dominated_count']}")

    print()
    print("=" * 80)
    print(f"REDUCED FACTORIAL: {len(combos)} configs")
    print("=" * 80)
    for c in combos[:10]:
        print(f"  {c['config_id']}")
    if len(combos) > 10:
        print(f"  ... and {len(combos) - 10} more")

    print()
    print("M4B COMPLETE — reduction.json written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())