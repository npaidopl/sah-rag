"""Aggregate all results into main tables (markdown + CSV + LaTeX)."""
import csv
import json
from pathlib import Path

from ablation.config import (
    ALL_DATASETS, STRUCTURE_RICH_DATASETS, EXCLUDE_FROM_MACRO,
    RESULTS_DIR, REPORTS_DIR,
)
from ablation.retrieval.variant_specs import VARIANT_SPECS


METRICS = ["recall@10", "recall@50", "ndcg@10", "hit@10", "mrr@10"]


def load_all_results() -> dict[tuple[str, str], dict]:
    """Load summary.json from every (variant, dataset) cell."""
    results = {}
    for summary_path in RESULTS_DIR.glob("*/*/summary.json"):
        try:
            data = json.loads(summary_path.read_text(encoding="utf-8"))
            key = (data["variant"], data["dataset"])
            results[key] = data
        except Exception as e:
            print(f"  WARN: cannot load {summary_path}: {e}")
    return results


def macro_average(results: dict, variant: str, exclude: set[str]) -> dict[str, float]:
    """Macro-average over (dataset, metric) pairs, excluding specified datasets."""
    sums = {m: [] for m in METRICS}
    for (v, ds), r in results.items():
        if v != variant or ds in exclude:
            continue
        for m in METRICS:
            if m in r:
                sums[m].append(r[m])
    return {m: sum(vs) / len(vs) if vs else 0.0 for m, vs in sums.items()}


def main_table(results: dict) -> str:
    """Main comparison table: variant × metric (macro over 15 non-excluded)."""
    lines = ["# Main comparison (macro over 15 datasets; m3docvqa/ruler excluded)", ""]
    header = f"{'variant':<30} " + " ".join(f"{m:>10}" for m in METRICS)
    lines.append(header)
    lines.append("-" * len(header))
    for v in sorted(VARIANT_SPECS.keys()):
        macro = macro_average(results, v, EXCLUDE_FROM_MACRO)
        if all(v_ == 0.0 for v_ in macro.values()):
            continue
        row = f"{v:<30} " + " ".join(f"{macro[m]:>10.4f}" for m in METRICS)
        lines.append(row)
    return "\n".join(lines) + "\n"


def per_dataset_table(results: dict) -> str:
    """Per-dataset recall@10 for the core variants (A4, A7, A1, A6, A9)."""
    core = ["A4_full_sah", "A7_paragraph_parents", "A1_flat_hybrid",
            "A6_boundaries_disabled", "A9_placebo_boundaries"]
    lines = ["# Per-dataset recall@10 (core structural variants)", ""]
    lines.append(f"{'dataset':<22} " + " ".join(f"{v:<18}" for v in core))
    lines.append("-" * (22 + 19 * len(core)))
    for ds in ALL_DATASETS:
        row = f"{ds:<22} "
        for v in core:
            r = results.get((v, ds), {}).get("recall@10")
            if r is None:
                cell = "-"
            else:
                cell = f"{r:.4f}"
            row += f"{cell:<18} "
        lines.append(row)
    return "\n".join(lines) + "\n"


def write_csv(results: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["variant", "dataset", "n_queries"] + METRICS)
        for (v, ds), r in sorted(results.items()):
            w.writerow([v, ds, r.get("n_queries", 0)] +
                       [r.get(m, "") for m in METRICS])


def main():
    results = load_all_results()
    print(f"Loaded {len(results)} result cells")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    main_md = main_table(results)
    (REPORTS_DIR / "main_table.md").write_text(main_md, encoding="utf-8")
    print(f"  {REPORTS_DIR / 'main_table.md'}")

    per_ds_md = per_dataset_table(results)
    (REPORTS_DIR / "per_dataset.md").write_text(per_ds_md, encoding="utf-8")
    print(f"  {REPORTS_DIR / 'per_dataset.md'}")

    write_csv(results, REPORTS_DIR / "all_results.csv")
    print(f"  {REPORTS_DIR / 'all_results.csv'}")

    print("\n" + main_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())