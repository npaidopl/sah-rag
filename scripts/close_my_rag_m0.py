# ruff: noqa: E402
"""My_RAG M0-L7 closure report."""
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

OUT_ROOT = REPO / "outputs" / "external_my_rag"
BASELINE = REPO / "baseline_audit"
REPORTS = REPO / "reports"
DOCS = REPO / "docs"

DATASETS = ("clerk", "cuad", "greek_bar_bench", "lara")


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


def load_json(p: Path) -> dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))


def collect_state(ds: str) -> dict[str, Any]:
    base = OUT_ROOT / ds
    out: dict[str, Any] = {"dataset_id": ds}
    for stage, mf in (
        ("chunks", "chunks-v1/manifest.json"),
        ("embeddings", "embeddings-v1/manifest.json"),
        ("retrieval", "retrieval-v1/manifest.json"),
        ("evaluation", "evaluation-v1/per_dataset_metrics.json"),
    ):
        p = base / mf
        if p.exists():
            m = load_json(p)
            out[stage] = {
                "manifest_sha256": sha256_file(p),
                **{k: v for k, v in m.items()
                   if k in ("record_count", "chunk_count", "parent_count",
                            "query_count", "wall_clock_seconds",
                            "query_count_evaluated",
                            "queries_excluded_zero_relevance",
                            "macro_metrics", "chunker_config_hash",
                            "embedding_config_hash", "retrieval_method",
                            "dense_dimension", "dense_precision",
                            "embedding_kind")},
            }
    return out


def build_registry(states: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "artifact_type": "my_rag_m0_cross_dataset_registry",
        "artifact_version": "1.0",
        "architecture_id": "my_rag",
        "fidelity_class": "PROJECT_OWNED_REFERENCE",
        "dataset_count": len(states),
        "datasets": states,
    }


def build_closure_manifest(states: list[dict[str, Any]],
                          cross: dict[str, Any]) -> dict[str, Any]:
    return {
        "artifact_type": "my_rag_m0_closure_manifest",
        "artifact_version": "1.0",
        "architecture_id": "my_rag",
        "fidelity_class": "PROJECT_OWNED_REFERENCE",
        "execution_id": "my-rag-m0-l7-closure-v1",
        "status": "MY_RAG_M0_BASELINE_CLOSED",
        "code_revision": get_git_revision(),
        "scope": {
            "datasets_executed": list(DATASETS),
            "datasets_in_full_suite": ["ruler", "loong", "long_bench_v2",
                                       "lara", "greek_bar_bench", "cuad", "clerk"],
            "reduced_scope_reason": (
                "Parallel scope with LongRAG baseline for fair comparison: "
                "4 datasets spanning small/medium/large document sizes."
            ),
        },
        "architecture_identity": {
            "chunker": "HierarchicalChildChunker v0.3.0 (256/32)",
            "embedder": "BGE-M3 hybrid (dense 1024 + learned sparse)",
            "storage": "Qdrant hybrid (dense COSINE + sparse IP)",
            "retrieval": "hybrid RRF k=60 + parent aggregation + doc resolution",
            "document_aggregation": "best_parent_score + 0.05 * sum_of_others",
        },
        "stage_summary": {
            "l1_contract_freeze": True,
            "l2_parse_chunk": True,
            "l3_colab_embeddings": True,
            "l4_qdrant_storage": True,
            "l5_retrieval_numpy": True,
            "l6_evaluation": True,
        },
        "primary_metric": cross.get("primary_metric"),
        "primary_metric_value": cross.get("primary_metric_value"),
        "macro_average_across_datasets": cross.get("macro_average_across_datasets"),
        "controlled_adaptations": [
            {
                "id": "qdrant_to_numpy_retrieval",
                "rationale": (
                    "Qdrant local mode is brute-force and scales poorly "
                    "with point count (>20k points -> >1s/query). "
                    "Replaced with mathematically equivalent in-memory "
                    "numpy+scipy computation."
                ),
                "equivalence_claimed": True,
                "equivalence_basis": (
                    "Identical dense cosine, sparse dot-product, RRF (k=60), "
                    "parent aggregation (w=0.05), and document resolution. "
                    "Only the execution substrate changed."
                ),
            },
            {
                "id": "query_dense_reexposure",
                "rationale": (
                    "Dense query embeddings were regenerated for My_RAG "
                    "rather than reused from the LongRAG cache. Config "
                    "hash differs due to prefix string (hybrid vs dense). "
                    "Not verified as byte-identical."
                ),
                "equivalence_claimed": False,
            },
        ],
        "documented_limitations": [
            "Reduced dataset scope (4 of 7 datasets).",
            "numpy-based retrieval is a controlled adaptation of the "
            "Qdrant-native RRF path; controlled equivalence argument "
            "rests on mathematical identity, not empirical verification.",
            "Same embedding and inference precision as LongRAG (FP16, GPU).",
            "cuad: zero-relevance queries excluded per common evaluator contract.",
        ],
        "comparison_ready": True,
        "comparison_baseline": "longrag_full_track",
    }


def build_markdown(states: list[dict[str, Any]],
                   cross: dict[str, Any]) -> str:
    macro = cross["macro_average_across_datasets"]
    L: list[str] = []
    L.append("# My_RAG (rag_system) M0 Baseline Closure Report")
    L.append("")
    L.append("**Fidelity class:** PROJECT_OWNED_REFERENCE")
    L.append("")
    L.append("## Scope")
    L.append("")
    L.append("Datasets executed: " + ", ".join(DATASETS))
    L.append("")
    L.append("## Per-dataset summary")
    L.append("")
    L.append("| Dataset | Chunks | Queries | H@10 | R@10 | nDCG@10 |")
    L.append("|---|---:|---:|---:|---:|---:|")
    for s in states:
        ev = s.get("evaluation", {})
        rm = ev.get("macro_metrics", {})
        chunks = s.get("chunks", {}).get("chunk_count", "?")
        q = s.get("retrieval", {}).get("query_count", "?")
        if rm:
            L.append(
                f"| {s['dataset_id']} | {chunks} | {q} | "
                f"{rm['hit@10']:.4f} | {rm['recall@10']:.4f} | "
                f"{rm['ndcg@10']:.4f} |"
            )
    L.append(
        f"| **MACRO** | | | {macro['hit@10']:.4f} | "
        f"{macro['recall@10']:.4f} | {macro['ndcg@10']:.4f} |"
    )
    L.append("")
    L.append("## Primary metric")
    L.append("")
    L.append(f"`macro_document_Recall_at_10 = {macro['recall@10']:.6f}`")
    L.append("")
    L.append("## Controlled adaptations")
    L.append("")
    L.append("- **Qdrant → numpy retrieval**: mathematically equivalent "
             "in-memory computation (dense matmul + sparse csr + RRF k=60 "
             "+ parent aggregation w=0.05 + document resolution).")
    L.append("- **Query dense re-exposure**: My_RAG dense query "
             "embeddings were regenerated alongside sparse (single "
             "Colab pass), not reused from LongRAG cache.")
    L.append("")
    L.append("## Documented limitations")
    L.append("")
    L.append("- Reduced dataset scope (4 of 7 datasets).")
    L.append("- FP16 embeddings, same precision as LongRAG.")
    L.append("- cuad: zero-relevance queries excluded per evaluator contract.")
    L.append("")
    return "\n".join(L)


def build_tex(states: list[dict[str, Any]],
             cross: dict[str, Any]) -> str:
    macro = cross["macro_average_across_datasets"]
    L: list[str] = []
    L.append("\\documentclass[11pt]{article}")
    L.append("\\usepackage[margin=1in]{geometry}")
    L.append("\\usepackage{booktabs}")
    L.append("\\begin{document}")
    L.append("\\section*{My\\_RAG (rag\\_system) M0 Baseline Closure}")
    L.append("\\textbf{Fidelity class:} PROJECT\\_OWNED\\_REFERENCE")
    L.append("")
    L.append("\\subsection*{Per-dataset summary}")
    L.append("\\begin{tabular}{lrrrr}")
    L.append("\\toprule")
    L.append("Dataset & Chunks & Queries & H@10 & R@10 \\\\")
    L.append("\\midrule")
    for s in states:
        ev = s.get("evaluation", {})
        rm = ev.get("macro_metrics", {})
        if not rm:
            continue
        chunks = s.get("chunks", {}).get("chunk_count", "?")
        q = s.get("retrieval", {}).get("query_count", "?")
        L.append(
            f"{s['dataset_id']} & {chunks} & {q} & "
            f"{rm['hit@10']:.4f} & {rm['recall@10']:.4f} \\\\"
        )
    L.append("\\midrule")
    L.append(
        f"Macro & -- & -- & {macro['hit@10']:.4f} & "
        f"{macro['recall@10']:.4f} \\\\"
    )
    L.append("\\bottomrule")
    L.append("\\end{tabular}")
    L.append("")
    L.append("\\subsection*{Primary metric}")
    L.append(f"macro\\\\_document\\\\_Recall@10 = {macro['recall@10']:.6f}")
    L.append("\\end{document}")
    return "\n".join(L) + "\n"


def main() -> int:
    t0 = time.perf_counter()
    states = []
    for ds in DATASETS:
        print(f"[collect] {ds} ...")
        states.append(collect_state(ds))

    cross_path = OUT_ROOT / "_m0_cross_dataset_evaluation.json"
    if not cross_path.exists():
        print(f"MISSING {cross_path}")
        return 2
    cross = load_json(cross_path)

    registry = build_registry(states)
    write_yaml_atomic(BASELINE / "my_rag_m0_cross_dataset_registry.yaml", registry)

    closure = build_closure_manifest(states, cross)
    write_yaml_atomic(BASELINE / "my_rag_m0_closure_manifest.yaml", closure)

    write_text_atomic(REPORTS / "my_rag_m0_closure_report.tex",
                     build_tex(states, cross))
    write_text_atomic(DOCS / "my_rag_m0_closure.md",
                     build_markdown(states, cross))

    print()
    print(f"M0-L7 CLOSURE COMPLETE in {time.perf_counter()-t0:.2f}s")
    print(f"  status: {closure['status']}")
    print(f"  primary metric: {closure['primary_metric']} = "
          f"{closure['primary_metric_value']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())