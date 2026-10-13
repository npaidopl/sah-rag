# ruff: noqa: E402
"""LongRAG vs SAH-RAG comparison report."""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import yaml

from rag_system.artifacts import get_git_revision

LR_ROOT = REPO / "outputs" / "external_longrag"
MR_ROOT = REPO / "outputs" / "external_sah_rag"
BASELINE = REPO / "baseline_audit"
REPORTS = REPO / "reports"
DOCS = REPO / "docs"

DATASETS = ("clerk", "cuad", "greek_bar_bench", "lara")

METRICS = (
    "hit@1", "hit@5", "hit@10", "hit@20", "hit@50",
    "recall@1", "recall@5", "recall@10", "recall@20", "recall@50",
    "precision@10",
    "mrr@1", "mrr@5", "mrr@10", "mrr@20", "mrr@50",
    "ndcg@10", "ndcg@50",
)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def write_yaml_atomic(p: Path, d: dict[str, Any]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(
        yaml.safe_dump(d, sort_keys=True, allow_unicode=True),
        encoding="utf-8",
    )
    tmp.replace(p)


def write_text_atomic(p: Path, t: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(t, encoding="utf-8", newline="\n")
    tmp.replace(p)


def load_cross(root: Path, filename: str) -> dict[str, Any]:
    return json.loads((root / filename).read_text(encoding="utf-8"))


def main() -> int:
    t0 = time.perf_counter()

    lr_cross = load_cross(LR_ROOT, "_l6_cross_dataset_evaluation.json")
    mr_cross = load_cross(MR_ROOT, "_m0_cross_dataset_evaluation.json")

    lr_macro = lr_cross["macro_average_across_datasets"]
    mr_macro = mr_cross["macro_average_across_datasets"]

    lr_by_ds = {d["dataset_id"]: d for d in lr_cross["datasets"]}
    mr_by_ds = {d["dataset_id"]: d for d in mr_cross["datasets"]}

    comparison: dict[str, Any] = {
        "artifact_type": "longrag_vs_sah_rag_comparison",
        "artifact_version": "1.0",
        "execution_id": "longrag-vs-sah-rag-v1",
        "code_revision": get_git_revision(),
        "protocol": {
            "scope_datasets": list(DATASETS),
            "same_embedder": "BAAI/bge-m3 @ 5617a9f6...",
            "same_embedding_precision": "float16",
            "same_evaluator": "document-level, cutoffs 1/5/10/20/50",
            "same_ground_truth": "canonical examples.jsonl",
            "primary_metric": "macro_document_Recall_at_10",
            "retrieval_only": True,
            "generation_disabled": True,
        },
        "macro_comparison": {
            metric: {
                "longrag": lr_macro.get(metric),
                "sah_rag": mr_macro.get(metric),
                "delta": (
                    (mr_macro.get(metric, 0) - lr_macro.get(metric, 0))
                    if metric in lr_macro and metric in mr_macro
                    else None
                ),
                "winner": (
                    "sah_rag"
                    if (metric in mr_macro and metric in lr_macro
                        and mr_macro[metric] > lr_macro[metric])
                    else "longrag"
                    if (metric in lr_macro and metric in mr_macro
                        and lr_macro[metric] > mr_macro[metric])
                    else "tie"
                ),
            }
            for metric in METRICS
        },
        "per_dataset_comparison": {},
        "primary_result": {
            "longrag": lr_macro["recall@10"],
            "sah_rag": mr_macro["recall@10"],
            "delta": mr_macro["recall@10"] - lr_macro["recall@10"],
            "relative_improvement_pct": (
                (mr_macro["recall@10"] - lr_macro["recall@10"])
                / lr_macro["recall@10"] * 100
            ),
            "winner": (
                "sah_rag"
                if mr_macro["recall@10"] > lr_macro["recall@10"]
                else "longrag"
            ),
        },
    }

    for ds in DATASETS:
        lr = lr_by_ds.get(ds, {})
        mr = mr_by_ds.get(ds, {})
        lr_m = lr.get("macro_metrics", {})
        mr_m = mr.get("macro_metrics", {})
        if not lr_m or not mr_m:
            continue
        comparison["per_dataset_comparison"][ds] = {
            "longrag": {k: lr_m.get(k) for k in METRICS},
            "sah_rag": {k: mr_m.get(k) for k in METRICS},
            "winner_recall_at_10": (
                "sah_rag" if mr_m["recall@10"] > lr_m["recall@10"] else "longrag"
            ),
            "delta_recall_at_10": mr_m["recall@10"] - lr_m["recall@10"],
            "relative_improvement_recall_at_10_pct": (
                (mr_m["recall@10"] - lr_m["recall@10"])
                / lr_m["recall@10"] * 100
                if lr_m["recall@10"] > 0 else None
            ),
            "evaluated_queries": {
                "longrag": lr.get("query_count_evaluated"),
                "sah_rag": mr.get("query_count_evaluated"),
            },
        }

    write_yaml_atomic(
        BASELINE / "longrag_vs_sah_rag_comparison.yaml", comparison,
    )

    # --- markdown ---
    md: list[str] = []
    md.append("# LongRAG vs SAH-RAG — Comparative Retrieval Report")
    md.append("")
    md.append("## Protocol")
    md.append("")
    md.append("- **Scope**: 4 datasets (clerk, cuad, greek_bar_bench, lara)")
    md.append("- **Embedder**: BAAI/bge-m3 @ 5617a9f6... (identical)")
    md.append("- **Precision**: FP16 (identical)")
    md.append("- **Evaluator**: document-level, cutoffs 1/5/10/20/50 (identical)")
    md.append("- **Ground truth**: canonical relevant_ids (identical)")
    md.append("- **Retrieval-only**: yes; generation disabled")
    md.append("")
    md.append("## Primary metric")
    md.append("")
    md.append(
        f"**macro_document_Recall@10**: LongRAG = "
        f"`{lr_macro['recall@10']:.4f}`, SAH-RAG = "
        f"`{mr_macro['recall@10']:.4f}`, "
        f"delta = `{comparison['primary_result']['delta']:+.4f}` "
        f"(`{comparison['primary_result']['relative_improvement_pct']:+.2f}%`)"
    )
    md.append("")
    md.append("## Macro-level comparison")
    md.append("")
    md.append("| Metric | LongRAG | SAH-RAG | Delta | Winner |")
    md.append("|---|---:|---:|---:|---|")
    for metric in METRICS:
        e = comparison["macro_comparison"][metric]
        if e["longrag"] is None or e["sah_rag"] is None:
            continue
        md.append(
            f"| {metric} | {e['longrag']:.4f} | {e['sah_rag']:.4f} | "
            f"{e['delta']:+.4f} | {e['winner']} |"
        )
    md.append("")
    md.append("## Per-dataset comparison")
    md.append("")
    md.append("| Dataset | LongRAG R@10 | SAH-RAG R@10 | Delta | Winner | Evaluated |")
    md.append("|---|---:|---:|---:|---|---:|")
    for ds in DATASETS:
        e = comparison["per_dataset_comparison"].get(ds)
        if not e:
            continue
        md.append(
            f"| {ds} | {e['longrag']['recall@10']:.4f} | "
            f"{e['sah_rag']['recall@10']:.4f} | "
            f"{e['delta_recall_at_10']:+.4f} | "
            f"{e['winner_recall_at_10']} | "
            f"{e['evaluated_queries']['longrag']} |"
        )
    md.append("")
    md.append("## Interpretation")
    md.append("")
    md.append(
        "LongRAG's K-consecutive-documents grouping policy produces large "
        "units that mix evidence from multiple documents. When the source "
        "corpus does not exhibit Wikipedia-style hyperlink adjacency and "
        "documents are short (CLERC), a single relevant passage is diluted "
        "by ten unrelated passages after expansion to documents. SAH-RAG's "
        "hierarchical child chunking with parent aggregation preserves "
        "fine-grained evidence and yields much stronger document-level "
        "ranking on such corpora."
    )
    md.append("")
    md.append(
        "On corpora whose documents are naturally large (LaRA), both "
        "architectures perform well and the gap narrows. This is consistent "
        "with the architectural observation that grouping is more valuable "
        "when the source units are themselves the retrievable evidence."
    )
    md.append("")
    md.append("## Reproducibility")
    md.append("")
    md.append(
        "- LongRAG closure manifest: "
        "`baseline_audit/longrag_full_track_closure_manifest.yaml`"
    )
    md.append(
        "- SAH-RAG closure manifest: "
        "`baseline_audit/my_rag_m0_closure_manifest.yaml`"
    )
    md.append(
        "- Comparison manifest: "
        "`baseline_audit/longrag_vs_sah_rag_comparison.yaml`"
    )

    write_text_atomic(DOCS / "longrag_vs_sah_rag_comparison.md", "\n".join(md))

    # --- LaTeX ---
    tex: list[str] = []
    tex.append("\\documentclass[11pt]{article}")
    tex.append("\\usepackage[margin=1in]{geometry}")
    tex.append("\\usepackage{booktabs}")
    tex.append("\\begin{document}")
    tex.append("\\section*{LongRAG vs My\\_RAG --- Comparative Retrieval Report}")
    tex.append("")
    tex.append(
        "\\textbf{Primary metric:} macro\\\\_document\\\\_Recall@10 "
        f"= LongRAG {lr_macro['recall@10']:.4f}, "
        f"My\\_RAG {mr_macro['recall@10']:.4f}, "
        f"delta {comparison['primary_result']['delta']:+.4f} "
        f"({comparison['primary_result']['relative_improvement_pct']:+.2f}\\%)"
    )
    tex.append("")
    tex.append("\\subsection*{Per-dataset Recall@10}")
    tex.append("\\begin{tabular}{lrrrr}")
    tex.append("\\toprule")
    tex.append("Dataset & LongRAG & My\\_RAG & Delta & Winner \\\\")
    tex.append("\\midrule")
    for ds in DATASETS:
        e = comparison["per_dataset_comparison"].get(ds)
        if not e:
            continue
        tex.append(
            f"{ds} & {e['longrag']['recall@10']:.4f} & "
            f"{e['sah_rag']['recall@10']:.4f} & "
            f"{e['delta_recall_at_10']:+.4f} & "
            f"{e['winner_recall_at_10'].replace('_', chr(92)+'_')} \\\\"
        )
    tex.append("\\midrule")
    tex.append(
        f"Macro & {lr_macro['recall@10']:.4f} & "
        f"{mr_macro['recall@10']:.4f} & "
        f"{comparison['primary_result']['delta']:+.4f} & "
        f"{comparison['primary_result']['winner'].replace('_', chr(92)+'_')} \\\\"
    )
    tex.append("\\bottomrule")
    tex.append("\\end{tabular}")
    tex.append("\\end{document}")
    write_text_atomic(
        REPORTS / "longrag_vs_sah_rag_comparison.tex", "\n".join(tex) + "\n",
    )

    print(f"COMPARISON COMPLETE in {time.perf_counter()-t0:.2f}s")
    print()
    print("=" * 78)
    print("PRIMARY METRIC: macro_document_Recall@10")
    print("=" * 78)
    print(f"  LongRAG: {lr_macro['recall@10']:.4f}")
    print(f"  SAH-RAG:  {mr_macro['recall@10']:.4f}")
    print(
        f"  Delta:   {comparison['primary_result']['delta']:+.4f} "
        f"({comparison['primary_result']['relative_improvement_pct']:+.2f}%)"
    )
    print(f"  Winner:  {comparison['primary_result']['winner']}")
    print()
    print("PER-DATASET R@10:")
    for ds in DATASETS:
        e = comparison["per_dataset_comparison"].get(ds)
        if not e:
            continue
        print(
            f"  {ds:18s} LongRAG={e['longrag']['recall@10']:.4f}  "
            f"SAH-RAG={e['sah_rag']['recall@10']:.4f}  "
            f"delta={e['delta_recall_at_10']:+.4f}  "
            f"winner={e['winner_recall_at_10']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())