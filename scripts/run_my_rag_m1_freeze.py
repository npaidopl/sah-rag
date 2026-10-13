# ruff: noqa: E402
"""My_RAG M1 - Freeze the optimization contract and dev/test split.

Deliverables:
  - baseline_audit/my_rag_optimization_contract.yaml
  - baseline_audit/my_rag_optimization_split.yaml
  - outputs/external_my_rag/_optimization_split.jsonl (gitignored)
  - outputs/external_my_rag/_optimization_split.manifest.json

The contract is frozen BEFORE any screening. The dev/test split is
grouped by document and deterministic with seed=0, per the program
handoff. No parameter tuning happens in M1; this stage only produces
the frozen inputs that all subsequent stages consume.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import yaml  # noqa: E402

from rag_system.canonical.loaders import (  # noqa: E402
    iter_canonical_examples,
)

DATASETS = ("clerk", "cuad", "greek_bar_bench", "lara")
SEED = 0
DEV_FRACTION = 0.5


# -----------------------------------------------------------------
# Source discovery
# -----------------------------------------------------------------
def discover_parameters() -> dict[str, Any]:
    """Source-verify the exposed parameter surface."""
    from rag_system.chunking.config import ChildChunkerConfig
    from rag_system.retrieval.qdrant import (
        retrieve_dense,
        retrieve_hybrid_rrf,
        retrieve_sparse,
    )

    out: dict[str, Any] = {}

    # --- chunker config ---
    try:
        c = ChildChunkerConfig()
        out["chunker_config_ok"] = True
        out["chunker_defaults"] = {
            "chunk_size": c.chunk_size,
            "chunk_overlap": c.chunk_overlap,
            "chunker_id": c.chunker_id,
            "chunker_version": c.chunker_version,
        }
        override_probes: dict[str, bool] = {}
        for cs, ov in [(512, 32), (1024, 128), (256, 0)]:
            try:
                ChildChunkerConfig(chunk_size=cs, chunk_overlap=ov)
                override_probes[f"{cs}_{ov}"] = True
            except Exception:
                override_probes[f"{cs}_{ov}"] = False
        out["chunker_override_ok"] = override_probes
    except Exception as exc:
        out["chunker_config_ok"] = False
        out["chunker_config_error"] = str(exc)

    # --- retrieval signatures ---
    for name, fn in (
        ("retrieve_dense", retrieve_dense),
        ("retrieve_sparse", retrieve_sparse),
        ("retrieve_hybrid_rrf", retrieve_hybrid_rrf),
    ):
        try:
            sig = inspect.signature(fn)
            out[f"{name}_params"] = list(sig.parameters.keys())
        except Exception as exc:
            out[f"{name}_error"] = str(exc)

    return out


# -----------------------------------------------------------------
# Dev/test split (grouped by document)
# -----------------------------------------------------------------
def build_split(
    dataset: str, *, seed: int = SEED, dev_fraction: float = DEV_FRACTION,
) -> dict[str, Any]:
    """Build a document-grouped, deterministic dev/test split."""
    doc_to_queries: dict[str, list[str]] = defaultdict(list)
    all_queries: list[str] = []

    for ex in iter_canonical_examples(dataset):
        qid = ex.query.query_id
        all_queries.append(qid)
        for did in ex.query.doc_ids:
            doc_to_queries[did].append(qid)

    all_docs = sorted(doc_to_queries.keys())
    rng = random.Random(seed)
    shuffled = all_docs[:]
    rng.shuffle(shuffled)

    total_q = len(all_queries)
    target_dev = int(total_q * dev_fraction)

    dev_docs: list[str] = []
    dev_qset: set[str] = set()
    accumulated = 0
    for did in shuffled:
        if accumulated >= target_dev and dev_docs:
            break
        dev_docs.append(did)
        for qid in doc_to_queries[did]:
            dev_qset.add(qid)
        accumulated += len(doc_to_queries[did])

    dev_docs_set = set(dev_docs)
    test_docs = sorted(set(all_docs) - dev_docs_set)
    test_qset = set(all_queries) - dev_qset

    return {
        "dataset": dataset,
        "total_queries": total_q,
        "total_docs": len(all_docs),
        "dev_doc_count": len(dev_docs),
        "test_doc_count": len(test_docs),
        "dev_query_count": len(dev_qset),
        "test_query_count": len(test_qset),
        "dev_query_fraction": len(dev_qset) / total_q if total_q else 0.0,
        "dev_doc_ids": sorted(dev_docs),
        "test_doc_ids": test_docs,
        "dev_query_ids": sorted(dev_qset),
        "test_query_ids": sorted(test_qset),
    }


def split_fingerprint(splits: dict[str, Any]) -> str:
    """Deterministic fingerprint over the query split assignment."""
    payload: dict[str, Any] = {}
    for ds, s in sorted(splits.items()):
        payload[ds] = {
            "dev_query_ids": s["dev_query_ids"],
            "test_query_ids": s["test_query_ids"],
        }
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# -----------------------------------------------------------------
# Writers
# -----------------------------------------------------------------
def write_yaml_atomic(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        yaml.safe_dump(data, sort_keys=True, allow_unicode=True),
        encoding="utf-8",
    )
    tmp.replace(path)


def write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def write_split_jsonl(path: Path, splits: dict[str, Any]) -> str:
    """Write one line per (dataset, query_id, split). Return SHA-256."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    h = hashlib.sha256()
    with tmp.open("w", encoding="utf-8", newline="\n") as fh:
        for ds in sorted(splits.keys()):
            s = splits[ds]
            for qid in s["dev_query_ids"]:
                line = json.dumps(
                    {"dataset": ds, "query_id": qid, "split": "dev"},
                    ensure_ascii=False, sort_keys=True,
                    separators=(",", ":"),
                ) + "\n"
                fh.write(line)
                h.update(line.encode("utf-8"))
            for qid in s["test_query_ids"]:
                line = json.dumps(
                    {"dataset": ds, "query_id": qid, "split": "test"},
                    ensure_ascii=False, sort_keys=True,
                    separators=(",", ":"),
                ) + "\n"
                fh.write(line)
                h.update(line.encode("utf-8"))
    tmp.replace(path)
    return h.hexdigest()


# -----------------------------------------------------------------
# Contract
# -----------------------------------------------------------------
def build_contract(
    discovery: dict[str, Any], split_fp: str,
) -> dict[str, Any]:
    return {
        "artifact_type": "my_rag_optimization_contract",
        "artifact_version": "1.0",
        "contract": {
            "id": "my_rag_m0_optimization",
            "version": "0.1.0",
            "status": "FROZEN_FOR_EXECUTION",
            "fidelity_class": "PROJECT_OWNED_REFERENCE",
        },
        "scope": {
            "datasets": list(DATASETS),
            "primary_metric": "macro_document_Recall_at_10",
            "evaluation_cutoffs": [1, 5, 10, 20, 50],
            "generation_disabled": True,
            "llm_judge_disabled": True,
        },
        "dev_test_split": {
            "policy": "grouped_by_document",
            "grouping_key": "query.doc_ids",
            "seed": SEED,
            "development_fraction": DEV_FRACTION,
            "test_fraction": 1.0 - DEV_FRACTION,
            "leakage_across_splits": "prohibited",
            "fingerprint_sha256": split_fp,
            "split_artifact": (
                "baseline_audit/my_rag_optimization_split.yaml"
            ),
            "split_jsonl": (
                "outputs/external_my_rag/_optimization_split.jsonl"
            ),
        },
        "source_discovery": discovery,
        "parameter_space": {
            "testable": [
                {
                    "name": "chunk_size",
                    "config_field": "ChildChunkerConfig.chunk_size",
                    "domain": [256, 512, 1024],
                    "role": "ARCHITECTURE_SUPPORTING",
                    "hypothesis": "H1",
                    "source_verified": True,
                },
                {
                    "name": "chunk_overlap",
                    "config_field": "ChildChunkerConfig.chunk_overlap",
                    "domain": [0, 32, 128],
                    "role": "ARCHITECTURE_SUPPORTING",
                    "hypothesis": "H2",
                    "source_verified": True,
                },
                {
                    "name": "retrieval_mode",
                    "config_field": (
                        "function choice: "
                        "retrieve_dense | retrieve_sparse | "
                        "retrieve_hybrid_rrf"
                    ),
                    "domain": ["dense", "sparse", "hybrid_rrf"],
                    "role": "ARCHITECTURE_SUPPORTING",
                    "hypothesis": "H5",
                    "source_verified": True,
                },
                {
                    "name": "candidate_limit",
                    "config_field": (
                        "retrieve_hybrid_rrf(candidate_limit, result_limit) "
                        "/ retrieve_dense(limit) / retrieve_sparse(limit)"
                    ),
                    "domain": [50, 100, 200],
                    "role": "ARCHITECTURE_SUPPORTING",
                    "hypothesis": "H3",
                    "source_verified": True,
                },
                {
                    "name": "final_top_k",
                    "config_field": "evaluation view; no new retrieval",
                    "domain": [5, 10, 20, 50],
                    "role": "EVALUATION_VIEW",
                    "hypothesis": "H4",
                    "source_verified": True,
                },
            ],
            "frozen": [
                {
                    "name": "parent_rest_weight",
                    "value": 0.05,
                    "reason": "architecture-defining aggregation formula",
                },
                {
                    "name": "rrf_k",
                    "value": 60,
                    "reason": (
                        "hardcoded constant in numpy path; "
                        "not exposed via Qdrant"
                    ),
                },
                {
                    "name": "preserve_parent_boundaries",
                    "value": True,
                    "reason": "pydantic validator enforces True",
                },
                {
                    "name": "structural_parser_policy",
                    "value": "project_default",
                    "reason": "architecture-defining",
                },
                {
                    "name": "document_resolution_policy",
                    "value": "best_parent_by_score",
                    "reason": "architecture-defining",
                },
                {
                    "name": "embedding_model_id",
                    "value": "BAAI/bge-m3",
                    "reason": "cross-architecture contract",
                },
                {
                    "name": "embedding_revision",
                    "value": (
                        "5617a9f61b028005a4858fdac845db406aefb181"
                    ),
                    "reason": "cross-architecture contract",
                },
                {
                    "name": "dense_dimension",
                    "value": 1024,
                    "reason": "model-fixed",
                },
                {
                    "name": "parent_limit",
                    "value": 100,
                    "reason": (
                        "evaluation design constraint; "
                        "sufficient for top-50 cutoffs"
                    ),
                },
                {
                    "name": "document_limit",
                    "value": 100,
                    "reason": (
                        "evaluation design constraint; "
                        "sufficient for top-50 cutoffs"
                    ),
                },
            ],
            "not_present": [
                {
                    "name": "rerank_depth",
                    "reason": (
                        "no reranker in pipeline; H7 not testable"
                    ),
                },
                {
                    "name": "maximum_context_tokens_per_query",
                    "reason": (
                        "no context builder; H6 not testable"
                    ),
                },
                {
                    "name": "row_context_token_cap",
                    "reason": (
                        "no context builder; H6 not testable"
                    ),
                },
            ],
        },
        "grid": {
            "chunking_variants": [
                {"chunk_size": 256, "chunk_overlap": 0},
                {"chunk_size": 256, "chunk_overlap": 32},
                {"chunk_size": 256, "chunk_overlap": 128},
                {"chunk_size": 512, "chunk_overlap": 0},
                {"chunk_size": 512, "chunk_overlap": 32},
                {"chunk_size": 512, "chunk_overlap": 128},
                {"chunk_size": 1024, "chunk_overlap": 0},
                {"chunk_size": 1024, "chunk_overlap": 32},
                {"chunk_size": 1024, "chunk_overlap": 128},
            ],
            "retrieval_modes": ["dense", "sparse", "hybrid_rrf"],
            "candidate_limits": [50, 100, 200],
            "evaluation_cutoffs": [1, 5, 10, 20, 50],
            "primary_metric": "macro_document_Recall_at_10",
        },
        "hypotheses": {
            "H1": {"status": "TESTABLE", "axis": "chunk_size"},
            "H2": {"status": "TESTABLE", "axis": "chunk_overlap"},
            "H3": {"status": "TESTABLE", "axis": "candidate_limit"},
            "H4": {
                "status": "TESTABLE_AS_EVALUATION_VIEW",
                "axis": "cutoffs",
            },
            "H5": {"status": "TESTABLE", "axis": "retrieval_mode"},
            "H6": {
                "status": "NOT_TESTABLE",
                "reason": "no context builder",
            },
            "H7": {
                "status": "NOT_TESTABLE",
                "reason": "no reranker",
            },
        },
        "decision_rules": {
            "monotonicity": {
                "direction_consistency": (
                    "at_least_80_percent_adjacent_steps"
                ),
                "total_change": "nonzero_with_bootstrap_support",
            },
            "saturation": {
                "marginal_quality_gain_threshold": 0.01,
            },
            "dominance": {
                "rule": (
                    "A_dominates_B_if_quality_ge_and_latency_le_"
                    "and_storage_le_and_at_least_one_strict"
                ),
            },
            "best_config_selection": {
                "rule": "one_standard_error",
                "tie_break": [
                    "lower_p95_latency",
                    "lower_storage_bytes",
                    "smaller_chunk_size",
                    "lower_candidate_k",
                    "config_id_ascending",
                ],
            },
        },
        "controlled_adaptations": [
            {
                "id": "query_embedding_reuse",
                "decision": (
                    "regenerate_dense_and_sparse_query_embeddings"
                ),
                "reason": "single_pass_simpler_than_partial_reuse",
            },
            {
                "id": "numpy_retrieval_substrate",
                "decision": "keep_numpy_based_retrieval",
                "reason": (
                    "mathematically_equivalent_to_qdrant_native_rrf"
                ),
            },
        ],
        "next_gate": "my_rag_m2_chunking_grid",
    }


# -----------------------------------------------------------------
# Main
# -----------------------------------------------------------------
def main() -> int:
    t0 = time.perf_counter()

    print("[M1] source discovery ...")
    discovery = discover_parameters()
    print(json.dumps(discovery, indent=2, default=str))
    print()

    print("[M1] building dev/test splits ...")
    splits: dict[str, Any] = {}
    for ds in DATASETS:
        s = build_split(ds, seed=SEED, dev_fraction=DEV_FRACTION)
        splits[ds] = s
        print(
            f"  {ds:18s} docs={s['total_docs']:>5} "
            f"dev_docs={s['dev_doc_count']:>5} "
            f"test_docs={s['test_doc_count']:>5} "
            f"dev_q={s['dev_query_count']:>6} "
            f"test_q={s['test_query_count']:>6} "
            f"frac={s['dev_query_fraction']:.3f}"
        )

    fp = split_fingerprint(splits)
    print(f"\n[M1] split fingerprint: {fp}")

    # split metadata YAML (small)
    split_doc: dict[str, Any] = {
        "artifact_type": "my_rag_optimization_split",
        "artifact_version": "1.0",
        "policy": "grouped_by_document",
        "grouping_key": "query.doc_ids",
        "seed": SEED,
        "development_fraction": DEV_FRACTION,
        "test_fraction": 1.0 - DEV_FRACTION,
        "leakage_guard": "document-level partition",
        "fingerprint_sha256": fp,
        "datasets": {
            ds: {
                "total_queries": s["total_queries"],
                "total_docs": s["total_docs"],
                "dev_doc_count": s["dev_doc_count"],
                "test_doc_count": s["test_doc_count"],
                "dev_query_count": s["dev_query_count"],
                "test_query_count": s["test_query_count"],
                "dev_query_fraction": s["dev_query_fraction"],
            }
            for ds, s in splits.items()
        },
        "full_partition_jsonl": (
            "outputs/external_my_rag/_optimization_split.jsonl"
        ),
    }
    split_path = (
        REPO / "baseline_audit" / "my_rag_optimization_split.yaml"
    )
    write_yaml_atomic(split_path, split_doc)
    print(f"[M1] wrote {split_path} ({split_path.stat().st_size} bytes)")

    # full partition JSONL (gitignored)
    out_root = REPO / "outputs" / "external_my_rag"
    out_root.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_root / "_optimization_split.jsonl"
    jsonl_sha = write_split_jsonl(jsonl_path, splits)
    write_json_atomic(
        out_root / "_optimization_split.manifest.json",
        {
            "artifact_type": "my_rag_optimization_split_jsonl",
            "artifact_version": "1.0",
            "record_count": sum(
                s["total_queries"] for s in splits.values()
            ),
            "sha256": jsonl_sha,
            "fingerprint_sha256": fp,
        },
    )
    print(
        f"[M1] wrote {jsonl_path} "
        f"({jsonl_path.stat().st_size} bytes, sha={jsonl_sha[:16]}...)"
    )

    # contract
    contract = build_contract(discovery, fp)
    contract_path = (
        REPO / "baseline_audit" / "my_rag_optimization_contract.yaml"
    )
    write_yaml_atomic(contract_path, contract)
    print(
        f"[M1] wrote {contract_path} "
        f"({contract_path.stat().st_size} bytes)"
    )

    print(f"\nM1 COMPLETE in {time.perf_counter()-t0:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())