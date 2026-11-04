"""Paired bootstrap comparison between variants (based on per-query metrics)."""
import json
from pathlib import Path

import numpy as np

from ablation.config import RESULTS_DIR, REPORTS_DIR, ALL_DATASETS, EXCLUDE_FROM_MACRO


def load_per_query(variant: str, dataset: str) -> dict[str, float]:
    """Load {query_id: recall@10} for one cell."""
    p = RESULTS_DIR / variant / dataset / "per_query_metrics.jsonl"
    if not p.is_file():
        return {}
    out = {}
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            out[r["query_id"]] = r.get("recall@10", 0.0)
    return out


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n_boot: int = 1000, seed: int = 42) -> dict:
    """Paired bootstrap on mean(b) - mean(a). Returns delta + 95% CI + p-value."""
    assert len(a) == len(b) and len(a) > 0
    rng = np.random.default_rng(seed)
    n = len(a)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        diffs[i] = b[idx].mean() - a[idx].mean()
    delta = b.mean() - a.mean()
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    # Two-sided p-value: fraction of bootstraps on opposite side of zero
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return {"delta": float(delta), "ci_lo": float(lo), "ci_hi": float(hi),
            "p_value": float(p), "n": n}


def compare_variants(v_a: str, v_b: str) -> dict:
    """Paired comparison across all non-excluded datasets, macro-averaged."""
    deltas = []
    per_ds = {}
    for ds in ALL_DATASETS:
        if ds in EXCLUDE_FROM_MACRO:
            continue
        ra = load_per_query(v_a, ds)
        rb = load_per_query(v_b, ds)
        common = sorted(set(ra) & set(rb))
        if not common:
            continue
        a = np.array([ra[q] for q in common])
        b = np.array([rb[q] for q in common])
        r = paired_bootstrap(a, b)
        per_ds[ds] = r
        deltas.append(r["delta"])
    if not deltas:
        return {"error": "no overlap"}
    return {
        "variant_a": v_a, "variant_b": v_b,
        "mean_delta_macro": float(np.mean(deltas)),
        "n_datasets": len(deltas),
        "per_dataset": per_ds,
    }


def main():
    comparisons = [
        ("A4_full_sah", "A1_flat_hybrid"),          # B24 decisive
        ("A4_full_sah", "A7_paragraph_parents"),    # universal structure
        ("A4_full_sah", "A6_boundaries_disabled"),  # boundary value
        ("A4_full_sah", "A9_placebo_boundaries"),   # semantic vs shape
        ("A4_full_sah", "A2_structure_dense_only"), # hybrid advantage
        ("A4_full_sah", "A3_structure_sparse_only"),# hybrid advantage
        ("A4_full_sah", "B1_best_only"),            # margin value
        ("A4_full_sah", "G4_weighted_5050"),        # RRF vs weighted
        ("A4_full_sah", "G8_random_sparse"),        # learned sparse value
        ("A4_full_sah", "D5_bm25_only"),            # vs BM25
    ]
    out = {}
    for a, b in comparisons:
        r = compare_variants(a, b)
        out[f"{a}__vs__{b}"] = r
        if "mean_delta_macro" in r:
            print(f"{a:<30} vs {b:<30}  Δ={r['mean_delta_macro']:+.4f}  "
                  f"over {r['n_datasets']} datasets")
        else:
            print(f"{a:<30} vs {b:<30}  {r.get('error', 'unknown')}")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "statistics.json").write_text(
        json.dumps(out, indent=2, default=float), encoding="utf-8")
    print(f"\nWrote {REPORTS_DIR / 'statistics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())