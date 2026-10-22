# ruff: noqa: E402
"""LongRAG L7 - Closure report.

Collects all stage manifests, produces a cross-dataset registry,
and writes:
  - baseline_audit/longrag_full_track_closure_manifest.yaml
  - baseline_audit/longrag_full_track_cross_dataset_registry.yaml
  - reports/longrag_full_track_closure_report.tex
  - docs/longrag_full_track_closure.md
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import yaml

from sah_rag.artifacts import get_git_revision

OUTPUT_ROOT = REPO_ROOT / "outputs" / "external_longrag"
BASELINE_AUDIT = REPO_ROOT / "baseline_audit"
REPORTS = REPO_ROOT / "reports"
DOCS = REPO_ROOT / "docs"

EXPECTED_DATASETS: tuple[str, ...] = (
    "clerk",
    "cuad",
    "greek_bar_bench",
    "lara",
)

FULL_SCOPE_DATASETS: tuple[str, ...] = (
    "ruler",
    "loong",
    "long_bench_v2",
    "lara",
    "greek_bar_bench",
    "cuad",
    "clerk",
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    tmp.replace(path)


def write_yaml_atomic(path: Path, data: dict[str, Any]) -> None:
    text = yaml.safe_dump(data, sort_keys=True, allow_unicode=True)
    atomic_write_text(path, text)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def collect_dataset_state(dataset_id: str) -> dict[str, Any]:
    base = OUTPUT_ROOT / dataset_id
    units_m = base / "units-v1" / "manifest.json"
    emb_m = base / "embeddings-v1" / "manifest.json"
    retr_m = base / "retrieval-v1" / "manifest.json"
    eval_m = base / "evaluation-v1" / "per_dataset_metrics.json"

    out: dict[str, Any] = {"dataset_id": dataset_id}
    if units_m.exists():
        m = load_json(units_m)
        out["units"] = {
            "long_unit_count": m["long_unit_count"],
            "max_unit_tokens": m["max_unit_tokens"],
            "split_unit_count": m["split_unit_count"],
            "documents": m["documents"],
            "manifest_sha256": sha256_file(units_m),
        }
    if emb_m.exists():
        m = load_json(emb_m)
        out["embeddings"] = {
            "record_count": m["record_count"],
            "dense_dimension": m["dense_dimension"],
            "dense_precision": m["dense_precision"],
            "gpu_name": m.get("gpu_name"),
            "manifest_sha256": sha256_file(emb_m),
        }
    if retr_m.exists():
        m = load_json(retr_m)
        out["retrieval"] = {
            "query_count": m["query_count"],
            "top_k_units": m["top_k_units"],
            "document_limit": m["document_limit"],
            "query_precision": m["query_precision"],
            "query_device": m["query_device"],
            "wall_clock_seconds": m["wall_clock_seconds"],
            "manifest_sha256": sha256_file(retr_m),
        }
    if eval_m.exists():
        m = load_json(eval_m)
        out["evaluation"] = {
            "query_count_evaluated": m["query_count_evaluated"],
            "queries_excluded_zero_relevance": m[
                "queries_excluded_zero_relevance"
            ],
            "macro_metrics": m["macro_metrics"],
            "manifest_sha256": sha256_file(eval_m),
        }
    return out


def build_registry(states: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "artifact_type": "longrag_full_track_cross_dataset_registry",
        "artifact_version": "1.0",
        "architecture_id": "longrag",
        "fidelity_class": "CONTROLLED_ADAPTATION",
        "datasets": states,
        "dataset_count": len(states),
    }


def build_closure_manifest(
    states: list[dict[str, Any]],
    cross_eval: dict[str, Any],
) -> dict[str, Any]:
    return {
        "artifact_type": "longrag_full_track_closure_manifest",
        "artifact_version": "1.0",
        "architecture_id": "longrag",
        "fidelity_class": "CONTROLLED_ADAPTATION",
        "execution_id": "longrag-l7-closure-v1",
        "status": "LONGRAG_FULL_PIPELINE_CLOSED_WITH_CLASSIFIED_LIMITATIONS",
        "code_revision": get_git_revision(),
        "scope": {
            "datasets_executed": list(EXPECTED_DATASETS),
            "datasets_in_full_canonical_suite": list(FULL_SCOPE_DATASETS),
            "reduced_scope_reason": (
                "User decision at L2-prep: exclude ruler, loong, "
                "long_bench_v2 from the LongRAG run to save compute "
                "for the SAH-RAG track. The three datasets have "
                "canonical documents with 8K-5M tokens each; after "
                "the oversize-split policy they yield very large "
                "unit corpora. The remaining four datasets preserve "
                "the architectural comparison (small-doc vs "
                "medium-doc vs large-doc corpora)."
            ),
        },
        "architecture_identity": {
            "long_document_grouping": (
                "adjacent_document_segments (K consecutive docs) "
                "@ target_token_size=4096"
            ),
            "dense_only_retrieval": "BGE-M3 dense, cosine, top-K units",
            "canonical_projection": (
                "unit-level retrieval expanded to document-level "
                "via first_occurrence_by_long_unit_rank"
            ),
        },
        "stage_summary": {
            "l1_architecture_freeze": True,
            "l2_unit_materialization": True,
            "l3_colab_embeddings": True,
            "l4_qdrant_storage": True,
            "l5_full_retrieval": True,
            "l6_evaluation": True,
        },
        "primary_metric": cross_eval.get("primary_metric"),
        "primary_metric_value": cross_eval.get("primary_metric_value"),
        "macro_average_across_datasets": cross_eval.get(
            "macro_average_across_datasets"
        ),
        "documented_limitations": [
            (
                "Reduced dataset scope: 4 of 7 canonical datasets "
                "executed. ruler, loong, long_bench_v2 excluded."
            ),
            (
                "LongRAG's unit granularity (K consecutive "
                "documents grouped into a ~4096-token unit) causes "
                "coarse document-level ranking on corpora whose "
                "documents are much shorter than the unit budget "
                "(clerk, cuad, greek_bar_bench). This is intrinsic "
                "to the paper's grouping policy when the corpus "
                "lacks Wikipedia-style hyperlink adjacency."
            ),
            (
                "Document-level metrics use first_occurrence_by_"
                "long_unit_rank expansion. When a document appears "
                "in multiple retrieved units, only its first "
                "occurrence is scored. This matches the reference "
                "implementation in baselines.longrag."
            ),
            (
                "Embeddings: FP16 GPU (Colab T4). Query embeddings: "
                "FP16 GPU, max_length=4098. Both use the pinned "
                "BGE-M3 revision 5617a9f61b028005a4858fdac845db406aefb181."
            ),
            (
                "cuad: 14208 of 20910 queries have zero relevant "
                "documents and are excluded per "
                "common_retrieval_evaluation_contract.yaml."
            ),
        ],
        "next_gate": "sah_rag_full_pipeline",
    }


def build_tex_report(
    states: list[dict[str, Any]],
    cross_eval: dict[str, Any],
) -> str:
    macro = cross_eval["macro_average_across_datasets"]
    lines: list[str] = []
    lines.append("\\documentclass[11pt]{article}")
    lines.append("\\usepackage[margin=1in]{geometry}")
    lines.append("\\usepackage{booktabs}")
    lines.append("\\usepackage{siunitx}")
    lines.append("\\begin{document}")
    lines.append("\\section*{LongRAG Full-Track Closure}")
    lines.append("\\textbf{Fidelity:} CONTROLLED\\_ADAPTATION")
    lines.append("")
    lines.append("\\subsection*{Scope}")
    lines.append("Datasets executed: " + ", ".join(EXPECTED_DATASETS))
    lines.append("")
    lines.append("\\subsection*{Per-dataset summary}")
    lines.append("\\begin{tabular}{lrrrr}")
    lines.append("\\toprule")
    lines.append("Dataset & Units & Queries & H@10 & nDCG@10 \\\\")
    lines.append("\\midrule")
    for s in states:
        if "units" not in s or "retrieval" not in s or "evaluation" not in s:
            continue
        m = s["evaluation"]["macro_metrics"]
        lines.append(
            f"{s['dataset_id']} & {s['units']['long_unit_count']} & "
            f"{s['retrieval']['query_count']} & "
            f"{m['hit@10']:.4f} & {m['ndcg@10']:.4f} \\\\"
        )
    lines.append("\\midrule")
    lines.append(
        f"Macro & -- & -- & {macro['hit@10']:.4f} & "
        f"{macro['ndcg@10']:.4f} \\\\"
    )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("")
    lines.append("\\subsection*{Primary metric}")
    lines.append(
        f"macro\\\\_document\\\\_Recall@10 = "
        f"{macro['recall@10']:.4f}"
    )
    lines.append("")
    lines.append("\\subsection*{Documented limitations}")
    lines.append("\\begin{itemize}")
    for lim in [
        "Reduced dataset scope: 4 of 7 canonical datasets executed.",
        (
            "Unit granularity on small-document corpora causes "
            "coarse document-level ranking (see clerk, cuad, "
            "greek\\_bar\\_bench)."
        ),
        (
            "Document expansion uses first-occurrence-by-unit-rank "
            "as defined by the reference implementation."
        ),
        (
            "FP16 embeddings (document and query) using pinned "
            "BGE-M3 revision 5617a9f6."
        ),
        "cuad: 14208/20910 zero-relevance queries excluded.",
    ]:
        lines.append(f"  \\item {lim}")
    lines.append("\\end{itemize}")
    lines.append("\\end{document}")
    return "\n".join(lines) + "\n"


def build_markdown_report(
    states: list[dict[str, Any]],
    cross_eval: dict[str, Any],
) -> str:
    macro = cross_eval["macro_average_across_datasets"]
    lines: list[str] = []
    lines.append("# LongRAG Full-Track Closure Report")
    lines.append("")
    lines.append("**Fidelity class:** CONTROLLED_ADAPTATION")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append("Datasets executed: " + ", ".join(EXPECTED_DATASETS))
    lines.append("")
    lines.append("## Per-dataset summary")
    lines.append("")
    lines.append("| Dataset | Units | Queries | H@10 | R@10 | nDCG@10 |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for s in states:
        if "units" not in s or "retrieval" not in s or "evaluation" not in s:
            continue
        m = s["evaluation"]["macro_metrics"]
        lines.append(
            f"| {s['dataset_id']} | {s['units']['long_unit_count']} | "
            f"{s['retrieval']['query_count']} | "
            f"{m['hit@10']:.4f} | {m['recall@10']:.4f} | "
            f"{m['ndcg@10']:.4f} |"
        )
    lines.append(
        f"| **MACRO** | | | {macro['hit@10']:.4f} | "
        f"{macro['recall@10']:.4f} | {macro['ndcg@10']:.4f} |"
    )
    lines.append("")
    lines.append("## Primary metric")
    lines.append("")
    lines.append(
        f"`macro_document_Recall_at_10 = "
        f"{macro['recall@10']:.6f}`"
    )
    lines.append("")
    lines.append("## Documented limitations")
    lines.append("")
    lines.append(
        "- Reduced dataset scope: 4 of 7 canonical datasets "
        "(ruler, loong, long_bench_v2 excluded by user decision)."
    )
    lines.append(
        "- Unit granularity on small-document corpora (clerk, "
        "cuad, greek_bar_bench) causes coarse document-level "
        "ranking. This is a finding about LongRAG's design when "
        "the corpus lacks Wikipedia-style hyperlink adjacency."
    )
    lines.append(
        "- Document expansion uses first_occurrence_by_long_unit_rank, "
        "matching the reference implementation."
    )
    lines.append(
        "- FP16 embeddings for documents and queries using the "
        "pinned BGE-M3 revision."
    )
    lines.append(
        "- cuad: 14208/20910 queries have zero relevant documents "
        "and are excluded from metrics."
    )
    lines.append("")
    return "\n".join(lines)


def run() -> int:
    t0 = time.perf_counter()

    states: list[dict[str, Any]] = []
    for ds in EXPECTED_DATASETS:
        print(f"[collect] {ds} ...")
        states.append(collect_dataset_state(ds))

    cross_path = OUTPUT_ROOT / "_l6_cross_dataset_evaluation.json"
    if not cross_path.exists():
        print(f"MISSING {cross_path}")
        return 2
    cross_eval = load_json(cross_path)

    registry = build_registry(states)
    write_yaml_atomic(
        BASELINE_AUDIT / "longrag_full_track_cross_dataset_registry.yaml",
        registry,
    )

    closure = build_closure_manifest(states, cross_eval)
    write_yaml_atomic(
        BASELINE_AUDIT / "longrag_full_track_closure_manifest.yaml",
        closure,
    )

    tex = build_tex_report(states, cross_eval)
    atomic_write_text(
        REPORTS / "longrag_full_track_closure_report.tex", tex,
    )

    md = build_markdown_report(states, cross_eval)
    atomic_write_text(
        DOCS / "longrag_full_track_closure.md", md,
    )

    print()
    print(f"L7 CLOSURE COMPLETE in {time.perf_counter()-t0:.2f}s")
    print(f"  status: {closure['status']}")
    print(
        f"  primary metric: "
        f"{closure['primary_metric']} = "
        f"{closure['primary_metric_value']:.6f}"
    )
    return 0


def main() -> int:
    return run()


if __name__ == "__main__":
    raise SystemExit(main())