# ruff: noqa: E402
"""LongRAG L1 - Architecture and dataset-envelope freeze.

Verifies the seven canonical datasets against the external registry,
loads the frozen LongRAG contracts, streams the canonical loaders, and
emits the L1 architecture freeze artifacts under baseline_audit/ and
outputs/external_longrag/_freeze-v1/.

Resume-safe: if the freeze manifest exists and is marked SUCCEEDED,
the script exits without re-running unless --force is passed.
"""
from __future__ import annotations
# --- SAH-RAG path resolution (sanitized from local absolute paths) ---
import os as _os
from pathlib import Path as _Path
_SAH_RAG_REPO_ROOT = _Path(__file__).resolve().parents[1]
_SAH_RAG_PROJECT_ROOT = _os.environ.get(
    "SAH_RAG_PROJECT_ROOT",
    str(_SAH_RAG_REPO_ROOT.parent / "Project 1"),
)
_SAH_RAG_CANONICAL_ROOT = _os.environ.get(
    "SAH_RAG_CANONICAL_ROOT",
    str(_Path(_SAH_RAG_PROJECT_ROOT) / "processed" / "canonical"),
)
# ---------------------------------------------------------------------

import argparse
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

import yaml  # noqa: E402

from rag_system.canonical.loaders import iter_validated_models  # noqa: E402
from rag_system.canonical.models import (  # noqa: E402
    CanonicalDocument,
    CanonicalExample,
)

CANONICAL_ROOT = Path(
    _SAH_RAG_CANONICAL_ROOT
)
BASELINE_AUDIT = REPO_ROOT / "baseline_audit"
OUTPUT_ROOT = REPO_ROOT / "outputs" / "external_longrag" / "_freeze-v1"

EXPECTED_DATASETS: tuple[str, ...] = (
    "ruler",
    "loong",
    "long_bench_v2",
    "lara",
    "greek_bar_bench",
    "cuad",
    "clerk",
)

EMBEDDING_MODEL_ID = "BAAI/bge-m3"
EMBEDDING_MODEL_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
EMBEDDING_SNAPSHOT_PATH = (
    "D:/rag_work/hf_cache/models--BAAI--bge-m3/snapshots/"
    "5617a9f61b028005a4858fdac845db406aefb181"
)

SOURCE_ARCHITECTURE_COMMIT = "ba92f95"

# L1a amendment (0.2.0): full-track long-unit semantics.
LONG_UNIT_TARGET_TOKEN_SIZE = 4096
LONG_UNIT_CONSTRUCTION_POLICY = "adjacent_document_segments"
LONG_UNIT_TOKEN_COUNTER = "pinned_bge_m3_tokenizer"
LONG_UNIT_OVERSIZE_HANDLING = (
    "split_into_target_size_windows_via_pinned_bge_m3_tokenizer"
)
CONTRACT_VERSION = "0.3.0"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
        .encode("utf-8")
        + b"\n"
    )


def atomic_write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(content)
    tmp.replace(path)


def write_yaml_atomic(path: Path, data: dict[str, Any]) -> None:
    text = yaml.safe_dump(data, sort_keys=True, allow_unicode=True)
    atomic_write_bytes(path, text.encode("utf-8"))


def write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    atomic_write_bytes(path, canonical_json_bytes(data))


def load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"Expected dict at {path}, got {type(data).__name__}")
    return data


def load_suite_registry() -> dict[str, Any]:
    return load_yaml(
        BASELINE_AUDIT / "external_canonical_suite_registry.yaml"
    )


def load_scope_contract() -> dict[str, Any]:
    return load_yaml(BASELINE_AUDIT / "dataset_corpus_scope_contract.yaml")


def load_projection_contract() -> dict[str, Any]:
    return load_yaml(
        BASELINE_AUDIT / "dataset_ground_truth_projection_contract.yaml"
    )


def load_frozen_longrag_contracts() -> dict[str, Any]:
    names = (
        "longrag_readiness.yaml",
        "longrag_readiness_declaration.yaml",
        "longrag_controlled_embedding_contract.yaml",
        "longrag_storage_equivalence_contract.yaml",
        "longrag_storage_equivalence_report.yaml",
        "longrag_retrieval_artifact_contract.yaml",
        "longrag_adapter_spec.yaml",
        "longrag_synthetic_acceptance_contract.yaml",
        "longrag_synthetic_acceptance_manifest.yaml",
        "longrag_environment_plan.yaml",
    )
    out: dict[str, Any] = {}
    for name in names:
        p = BASELINE_AUDIT / name
        if not p.exists():
            raise FileNotFoundError(f"Missing frozen contract: {p}")
        out[name] = load_yaml(p)
    return out


def verify_dataset_shas(
    registry: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    datasets = registry.get("datasets", {})
    for ds_id in EXPECTED_DATASETS:
        entry = datasets.get(ds_id)
        if entry is None:
            raise KeyError(f"Dataset {ds_id!r} missing from registry.")
        files = entry["files"]
        doc_meta = files["documents.jsonl"]
        ex_meta = files["examples.jsonl"]
        doc_path = Path(doc_meta["path"])
        ex_path = Path(ex_meta["path"])
        actual_doc_sha = sha256_file(doc_path)
        actual_ex_sha = sha256_file(ex_path)
        if actual_doc_sha != doc_meta["sha256"]:
            raise RuntimeError(
                f"SHA mismatch documents.jsonl {ds_id}: "
                f"{actual_doc_sha} != {doc_meta['sha256']}"
            )
        if actual_ex_sha != ex_meta["sha256"]:
            raise RuntimeError(
                f"SHA mismatch examples.jsonl {ds_id}: "
                f"{actual_ex_sha} != {ex_meta['sha256']}"
            )
        results[ds_id] = {
            "documents_sha256": actual_doc_sha,
            "examples_sha256": actual_ex_sha,
            "documents_bytes": doc_path.stat().st_size,
            "examples_bytes": ex_path.stat().st_size,
        }
    return results


def stream_and_count(
    doc_path: Path,
    examples_path: Path,
) -> dict[str, int]:
    n_docs = 0
    for _ in iter_validated_models(doc_path, CanonicalDocument):
        n_docs += 1
    n_examples = 0
    for _ in iter_validated_models(examples_path, CanonicalExample):
        n_examples += 1
    return {"documents": n_docs, "examples": n_examples}


def stream_all_datasets(
    registry: dict[str, Any],
) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for ds_id in EXPECTED_DATASETS:
        files = registry["datasets"][ds_id]["files"]
        doc_path = Path(files["documents.jsonl"]["path"])
        ex_path = Path(files["examples.jsonl"]["path"])
        out[ds_id] = stream_and_count(doc_path, ex_path)
    return out


def build_embedding_profile() -> dict[str, Any]:
    return {
        "artifact_type": "longrag_full_track_embedding_profile",
        "artifact_version": "1.0",
        "contract": {
            "id": "longrag_full_track_bge_m3_dense",
            "version": "0.3.0",
            "status": "FROZEN_FOR_EXECUTION",
            "scope": "seven_canonical_datasets_full_track",
            "supersedes_within_full_track_only": (
                "longrag_bge_m3_dense_only"
            ),
            "synthetic_acceptance_contract_untouched": True,
            "rationale": (
                "The synthetic-acceptance embedding contract was "
                "scoped to a 12-document fixture with CPU FP32. The "
                "full-track execution is Colab GPU with FP16 primary "
                "(autonomous handoff L3). This is a new contract, "
                "not a modification of the synthetic contract."
            ),
        },
        "model": {
            "provider": "HuggingFace",
            "library": "FlagEmbedding",
            "class_name": "BGEM3FlagModel",
            "model_id": EMBEDDING_MODEL_ID,
            "model_revision": EMBEDDING_MODEL_REVISION,
            "local_snapshot_path": EMBEDDING_SNAPSHOT_PATH,
            "local_snapshot_required": True,
            "network_fallback_allowed": False,
        },
        "execution": {
            "environment": "Google Colab GPU",
            "device": "cuda",
            "dtype_primary": "float16",
            "batch_size_primary": 16,
            "dense_enabled": True,
            "sparse_enabled": False,
            "multi_vector_enabled": False,
            "dense_dimension": 1024,
            "dense_normalized": True,
            "pooling": "CLS",
        },
        "bounded_reference_profile": {
            "enabled": True,
            "environment": "local CPU",
            "device": "cpu",
            "dtype": "float32",
            "batch_size": 2,
            "subset_size_per_dataset": 100,
            "purpose": (
                "Bounded precision equivalence check: FP16-GPU vs "
                "FP32-CPU on a small stratified subset, confirming "
                "FP16 does not alter top-k identity."
            ),
        },
        "safety": {
            "model_download_enabled": False,
            "network_access_enabled": False,
            "hosted_requests_enabled": False,
        },
    }


def build_architecture_freeze() -> dict[str, Any]:
    return {
        "artifact_type": (
            "longrag_full_track_architecture_freeze_contract"
        ),
        "artifact_version": "1.0",
        "contract": {
            "id": "longrag_full_track_architecture_freeze",
            "version": "0.3.0",
            "status": "FROZEN_FOR_EXECUTION",
        },
        "architecture": {
            "architecture_id": "longrag",
            "display_name": "LongRAG",
            "fidelity_class": "CONTROLLED_ADAPTATION",
            "controlled_track_eligible": True,
            "native_track_eligible": False,
            "native_track_reason": "native_embedder_identity_unpinned",
            "source_architecture_commit": SOURCE_ARCHITECTURE_COMMIT,
        },
        "architecture_defining_components": [
            "long_document_grouping",
            "dense_only_retrieval",
        ],
        "long_unit_semantics": {
            "construction_policy": LONG_UNIT_CONSTRUCTION_POLICY,
            "target_token_size": LONG_UNIT_TARGET_TOKEN_SIZE,
            "token_counter": LONG_UNIT_TOKEN_COUNTER,
            "segment_source": "one_canonical_document_per_segment",
            "derived_parent_id": "<doc_id>:p0",
            "derived_chunk_id": "seg-<doc_id>",
            "oversize_handling": LONG_UNIT_OVERSIZE_HANDLING,
            "tokenizer_id": EMBEDDING_MODEL_ID,
            "tokenizer_revision": EMBEDDING_MODEL_REVISION,
            "long_unit_identity_property": (
                "deterministic_content_addressable"
            ),
        },
        "retrieval_semantics": {
            "retrieval_level": "long_unit",
            "retrieval_signal": "dense_only",
            "similarity": "COSINE",
            "score_direction": "HIGHER_IS_BETTER",
            "reranking_enabled": False,
            "hybrid_enabled": False,
            "sparse_enabled": False,
            "expansion": "first_occurrence_by_long_unit_rank",
        },
        "storage_semantics": {
            "backend": "Qdrant",
            "mode": "local_persistence",
            "vector_name": "dense",
            "dimension": 1024,
            "distance": "COSINE",
            "point_id_prefix": "rag-longunit",
            "point_id_namespace": "uuid.NAMESPACE_URL",
            "approximate_search_allowed": False,
            "quantization_allowed": False,
            "equivalence_classification": (
                "SEMANTICALLY_NEUTRAL_FOR_CONTROLLED_TRACK"
            ),
            "equivalence_evidence": (
                "baseline_audit/"
                "longrag_storage_equivalence_report.yaml"
            ),
        },
        "evaluation_profile": {
            "level": "canonical_document",
            "metrics": [
                "Hit@1",
                "Hit@5",
                "Hit@10",
                "Hit@20",
                "Hit@50",
                "Recall@1",
                "Recall@5",
                "Recall@10",
                "Recall@20",
                "Recall@50",
                "Precision@10",
                "MRR@10",
                "MRR@50",
                "nDCG@10",
                "nDCG@50",
            ],
            "primary_metric": "macro_document_Recall_at_10",
            "aggregation": "macro_then_micro_secondary",
            "ground_truth_boundary": "evaluator_only",
            "generation_enabled": False,
            "llm_judge_enabled": False,
        },
        "datasets": list(EXPECTED_DATASETS),
        "embedding_profile_ref": (
            "longrag_full_track_embedding_profile"
        ),
        "component_disposition_ref": (
            "longrag_full_track_component_disposition"
        ),
    }


def build_component_disposition() -> dict[str, Any]:
    return {
        "artifact_type": "longrag_full_track_component_disposition",
        "artifact_version": "1.0",
        "architecture_id": "longrag",
        "fidelity_class": "CONTROLLED_ADAPTATION",
        "components": {
            "long_document_grouping": {
                "role": "ARCHITECTURE_DEFINING",
                "treatment": "PRESERVED",
                "justification": (
                    "LongRAG's defining characteristic is grouping "
                    "adjacent segments into long retrieval units. "
                    "Replacing with flat chunks would collapse the "
                    "architecture into a different design."
                ),
                "fidelity_risk": "NONE",
                "validation": (
                    "deterministic; non-empty member_doc_ids; one "
                    "unit per document minimum; adjacent consecutive "
                    "segments packed to target budget."
                ),
            },
            "dense_only_retrieval": {
                "role": "ARCHITECTURE_DEFINING",
                "treatment": "PRESERVED",
                "justification": (
                    "LongRAG retrieves long units via dense "
                    "similarity over a reduced-unit corpus. "
                    "Introducing sparse, hybrid, or reranked "
                    "retrieval would change the architecture."
                ),
                "fidelity_risk": "NONE",
                "validation": (
                    "single dense vector per unit; COSINE "
                    "distance; no sparse signal; no reranking; no "
                    "hybrid fusion."
                ),
            },
            "canonical_document_input": {
                "role": "ARCHITECTURE_SUPPORTING",
                "treatment": "CONTROLLED_ADAPTATION",
                "justification": (
                    "LongRAG's original input is Wikipedia "
                    "hyperlink grouping. Canonical documents "
                    "replace that per "
                    "baseline_audit/longrag_adapter_spec.yaml."
                ),
                "fidelity_risk": "LOW",
                "validation": (
                    "deterministic build_source_segments from "
                    "CanonicalDocument; 1 document = 1 source "
                    "segment."
                ),
            },
            "embedder": {
                "role": "ARCHITECTURE_SUPPORTING",
                "treatment": "CONTROLLED_REPLACEMENT",
                "justification": (
                    "BGE-M3 dense-only as controlled encoder per "
                    "longrag_controlled_embedding_contract.yaml. "
                    "Native LongRAG embedder identity is unpinned "
                    "and equivalence is not claimed."
                ),
                "fidelity_risk": "MEDIUM",
                "validation": (
                    "pinned revision "
                    "5617a9f61b028005a4858fdac845db406aefb181; "
                    "dense normalized; dimension 1024."
                ),
            },
            "tokenizer": {
                "role": "ARCHITECTURE_SUPPORTING",
                "treatment": "MATCHES_EMBEDDER",
                "justification": (
                    "Tokenizer identity must match embedder "
                    "identity per "
                    "component_substitution_policy.yaml."
                ),
                "fidelity_risk": "NONE",
                "validation": (
                    "BAAI/bge-m3 revision "
                    "5617a9f61b028005a4858fdac845db406aefb181."
                ),
            },
            "storage_backend": {
                "role": "ARCHITECTURE_SUPPORTING",
                "treatment": "CONTROLLED_REPLACEMENT",
                "justification": (
                    "Qdrant dense-only with exact equivalence "
                    "demonstrated on the frozen synthetic fixture "
                    "(longrag_storage_equivalence_report.yaml)."
                ),
                "fidelity_risk": "LOW",
                "validation": (
                    "SEMANTICALLY_NEUTRAL_FOR_CONTROLLED_TRACK; "
                    "top-1, top-5, top-10 overlap = 1.0 on fixture."
                ),
            },
            "llm_index_time": {
                "role": "NOT_PRESENT",
                "treatment": "N/A",
                "justification": (
                    "LongRAG does not require an LLM at index time."
                ),
                "fidelity_risk": "NONE",
                "validation": "not applicable.",
            },
            "llm_query_time": {
                "role": "NOT_PRESENT",
                "treatment": "N/A",
                "justification": (
                    "LongRAG retrieval is dense-only; no query-time "
                    "LLM stage is part of the architecture."
                ),
                "fidelity_risk": "NONE",
                "validation": "not applicable.",
            },
            "llm_final_generation": {
                "role": "OPTIONAL_DOWNSTREAM",
                "treatment": "DISABLED",
                "justification": (
                    "Retrieval-only benchmark; generation is out "
                    "of scope per "
                    "common_retrieval_evaluation_contract.yaml."
                ),
                "fidelity_risk": "NONE",
                "validation": "disabled in all L1-L7 stages.",
            },
            "long_reader": {
                "role": "OPTIONAL_DOWNSTREAM",
                "treatment": "DISABLED",
                "justification": (
                    "LongRAG's long reader is an answer-synthesis "
                    "stage and is explicitly excluded in "
                    "longrag_environment_plan.yaml."
                ),
                "fidelity_risk": "NONE",
                "validation": "disabled in all L1-L7 stages.",
            },
            "oversize_handling": {
                "role": "ARCHITECTURE_SUPPORTING",
                "treatment": "CONTROLLED_ADAPTATION",
                "justification": (
                    "Long-context canonical documents can exceed "
                    "any finite target unit size. The source-"
                    "corrected pilot pattern splits oversized "
                    "segments into target-size windows using "
                    "the pinned BGE-M3 tokenizer. This does not "
                    "alter LongRAG's long-unit retrieval "
                    "semantics; it preserves complete source "
                    "coverage under a fixed budget."
                ),
                "fidelity_risk": "LOW",
                "validation": (
                    "deterministic window boundaries; every "
                    "window decoded with pinned tokenizer; "
                    "member_doc_ids preserved; complete_source_"
                    "coverage enforced by summarize_long_units."
                ),
            },
            "tokenizer_bound_segment_tokens": {
                "role": "ARCHITECTURE_SUPPORTING",
                "treatment": "MATCHES_EMBEDDER",
                "justification": (
                    "Segment token counts must be measured with "
                    "the same tokenizer that will produce "
                    "embeddings, otherwise the 4096-token budget "
                    "does not correspond to the model input "
                    "budget. The source-corrected pilot already "
                    "adopted this."
                ),
                "fidelity_risk": "NONE",
                "validation": (
                    "pinned BGE-M3 revision "
                    "5617a9f61b028005a4858fdac845db406aefb181; "
                    "add_special_tokens=False for measurement, "
                    "special_token_count=2 accounted at "
                    "embedding."
                ),
            },
        },
    }


def build_dataset_envelope(
    registry: dict[str, Any],
    scope: dict[str, Any],
    projection: dict[str, Any],
    sha_results: dict[str, dict[str, Any]],
    loader_results: dict[str, dict[str, int]],
) -> dict[str, Any]:
    datasets: dict[str, Any] = {}
    scope_datasets = scope["datasets"]
    proj_datasets = projection["datasets"]
    reg_datasets = registry["datasets"]
    for ds_id in EXPECTED_DATASETS:
        reg = reg_datasets[ds_id]
        sc = scope_datasets[ds_id]
        pj = proj_datasets[ds_id]
        shas = sha_results[ds_id]
        loaded = loader_results.get(ds_id, {})
        datasets[ds_id] = {
            "canonical_root": reg["canonical_root"],
            "documents_sha256": shas["documents_sha256"],
            "examples_sha256": shas["examples_sha256"],
            "documents_bytes": shas["documents_bytes"],
            "examples_bytes": shas["examples_bytes"],
            "documents_count_registry": reg["manifest"]["documents"],
            "examples_count_registry": reg["manifest"]["examples"],
            "documents_count_streamed": loaded.get("documents"),
            "examples_count_streamed": loaded.get("examples"),
            "retrieval_scope": sc["expected_scope"],
            "corpus_selector": sc["corpus_selector"],
            "primary_evaluation_unit": pj["primary_evaluation_unit"],
            "secondary_evaluation_units": pj[
                "secondary_evaluation_units"
            ],
            "document_ids_from": pj["document_ids_from"],
            "evidence_ids_from": pj["evidence_ids_from"],
            "spans_from": pj["spans_from"],
            "answers_from": pj["answers_from"],
            "ground_truth_used_during_indexing": pj[
                "ground_truth_used_during_indexing"
            ],
            "ground_truth_used_during_querying": pj[
                "ground_truth_used_during_querying"
            ],
            "splits_registry": reg["manifest"].get("splits", {}),
            "output_root": f"outputs/external_longrag/{ds_id}",
            "qdrant_collection_name": f"longrag-{ds_id}-v1",
        }
    return {
        "artifact_type": "longrag_full_track_dataset_envelope",
        "artifact_version": "1.0",
        "contract": {
            "id": "longrag_full_track_dataset_envelope",
            "version": "0.3.0",
            "status": "FROZEN_FOR_EXECUTION",
            "source_contracts": [
                (
                    "baseline_audit/"
                    "external_canonical_suite_registry.yaml"
                ),
                (
                    "baseline_audit/"
                    "dataset_corpus_scope_contract.yaml"
                ),
                (
                    "baseline_audit/"
                    "dataset_ground_truth_projection_contract.yaml"
                ),
            ],
        },
        "dataset_count": len(EXPECTED_DATASETS),
        "datasets": datasets,
        "totals": {
            "documents": sum(
                d["documents_count_registry"]
                for d in datasets.values()
            ),
            "examples": sum(
                d["examples_count_registry"]
                for d in datasets.values()
            ),
        },
    }


def build_plan() -> dict[str, Any]:
    return {
        "artifact_type": "longrag_full_track_plan",
        "artifact_version": "1.0",
        "architecture_id": "longrag",
        "order": ["L1", "L2", "L3", "L4", "L5", "L6", "L7"],
        "phases": {
            "L1": {
                "name": "architecture_and_dataset_envelope_freeze",
                "script": (
                    "scripts/"
                    "run_longrag_architecture_freeze_batch.py"
                ),
                "inputs": [
                    (
                        "baseline_audit/"
                        "external_canonical_suite_registry.yaml"
                    ),
                    "baseline_audit/longrag_*.yaml",
                    "src/rag_system/canonical/",
                    (
                        _SAH_RAG_CANONICAL_ROOT,
                    ),
                ],
                "outputs": [
                    (
                        "baseline_audit/"
                        "longrag_full_track_embedding_profile.yaml"
                    ),
                    (
                        "baseline_audit/"
                        "longrag_full_track_architecture_freeze_"
                        "contract.yaml"
                    ),
                    (
                        "baseline_audit/"
                        "longrag_full_track_component_"
                        "disposition.yaml"
                    ),
                    (
                        "baseline_audit/"
                        "longrag_full_track_dataset_envelope.yaml"
                    ),
                    (
                        "baseline_audit/"
                        "longrag_full_track_plan.yaml"
                    ),
                    (
                        "outputs/external_longrag/_freeze-v1/"
                        "verification_report.json"
                    ),
                    (
                        "outputs/external_longrag/_freeze-v1/"
                        "manifest.json"
                    ),
                    (
                        "outputs/external_longrag/_freeze-v1/"
                        "checksums.json"
                    ),
                ],
                "acceptance": {
                    "canonical_datasets_valid": 7,
                    "embedding_model_pinned": True,
                    "embedding_revision_pinned": True,
                    "retrieval_policy_frozen": True,
                    "evaluation_mapping_defined": True,
                },
                "abort_conditions": [
                    "canonical_sha256_mismatch",
                    "missing_frozen_contract",
                    "canonical_model_loading_failure",
                    "dataset_count_mismatch",
                ],
            },
            "L2": {
                "name": "unit_materialization",
                "script": (
                    "scripts/"
                    "run_longrag_unit_materialization_batch.py"
                ),
                "depends_on": ["L1"],
                "per_dataset_output": (
                    "outputs/external_longrag/<dataset>/units-v1/"
                ),
                "acceptance": {
                    "complete_source_coverage": True,
                    "deterministic_unit_ids": True,
                    "source_immutability": True,
                    "lineage_closure": True,
                },
            },
            "L3": {
                "name": "colab_gpu_embeddings",
                "prepare_script": (
                    "scripts/"
                    "prepare_longrag_colab_embedding_batch.py"
                ),
                "import_script": (
                    "scripts/"
                    "import_longrag_colab_embedding_batch.py"
                ),
                "depends_on": ["L2"],
                "environment": "Google Colab GPU",
                "transfer": [
                    "longrag-embedding-input-v1.tar.gz",
                    "longrag-embedding-output-v1.tar.gz",
                ],
                "primary_precision": "float16",
                "bounded_reference_precision": "float32",
            },
            "L4": {
                "name": "storage_build_and_equivalence",
                "script": (
                    "scripts/"
                    "run_longrag_storage_and_parity_batch.py"
                ),
                "depends_on": ["L3"],
                "backend": "Qdrant",
                "collection_pattern": "longrag-<dataset>-v1",
                "reopen_required": True,
            },
            "L5": {
                "name": "full_retrieval",
                "script": (
                    "scripts/run_longrag_full_retrieval_batch.py"
                ),
                "depends_on": ["L4"],
                "query_projection": ["query_id", "query_text"],
                "ground_truth_visible": False,
            },
            "L6": {
                "name": "ground_truth_evaluation",
                "script": (
                    "scripts/run_longrag_full_evaluation_batch.py"
                ),
                "depends_on": ["L5"],
                "evaluator_isolation": True,
                "metrics": "see architecture_freeze.evaluation_profile",
            },
            "L7": {
                "name": "closure",
                "script": "scripts/close_longrag_full_track.py",
                "depends_on": ["L6"],
            },
        },
    }


def compute_checksums(paths: dict[str, Path]) -> dict[str, str]:
    out: dict[str, str] = {}
    for key, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Cannot checksum missing: {path}")
        out[key] = sha256_file(path)
    return out


def build_verification_report(
    sha_results: dict[str, dict[str, Any]],
    loader_results: dict[str, dict[str, int]],
    registry: dict[str, Any],
) -> dict[str, Any]:
    return {
        "artifact_type": (
            "longrag_full_track_architecture_freeze_verification_report"
        ),
        "artifact_version": "1.0",
        "status": "PASSED",
        "dataset_count": len(EXPECTED_DATASETS),
        "datasets": list(EXPECTED_DATASETS),
        "canonical_registry_fingerprint": registry["suite"][
            "collection_fingerprint"
        ],
        "sha256_checks": sha_results,
        "streamed_counts": loader_results,
    }


def is_complete(manifest_path: Path) -> bool:
    if not manifest_path.exists():
        return False
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return bool(data.get("status") == "SUCCEEDED")


def run(*, force: bool, skip_streaming: bool) -> int:
    t0 = time.perf_counter()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    manifest_path = OUTPUT_ROOT / "manifest.json"

    if not force and is_complete(manifest_path):
        print(
            "[resume] L1 manifest already SUCCEEDED; "
            "use --force to re-run."
        )
        return 0

    print("[1/8] loading frozen contracts ...")
    load_frozen_longrag_contracts()
    registry = load_suite_registry()
    scope = load_scope_contract()
    projection = load_projection_contract()

    print("[2/8] verifying canonical SHA-256 ...")
    sha_results = verify_dataset_shas(registry)
    for ds_id, sha in sha_results.items():
        print(
            f"      {ds_id:18s} "
            f"documents={sha['documents_sha256'][:12]} "
            f"examples={sha['examples_sha256'][:12]}"
        )

    loader_results: dict[str, dict[str, int]] = {}
    if skip_streaming:
        print("[3/8] streaming skipped (--skip-streaming)")
    else:
        print("[3/8] streaming canonical loaders ...")
        loader_results = stream_all_datasets(registry)
        for ds_id, counts in loader_results.items():
            print(
                f"      {ds_id:18s} "
                f"documents={counts['documents']} "
                f"examples={counts['examples']}"
            )

    print("[4/8] building L1 artifacts ...")
    embedding_profile = build_embedding_profile()
    architecture_freeze = build_architecture_freeze()
    disposition = build_component_disposition()
    envelope = build_dataset_envelope(
        registry,
        scope,
        projection,
        sha_results,
        loader_results,
    )
    plan = build_plan()

    print("[5/8] writing baseline_audit YAMLs ...")
    p_embedding = (
        BASELINE_AUDIT / "longrag_full_track_embedding_profile.yaml"
    )
    p_freeze = (
        BASELINE_AUDIT
        / "longrag_full_track_architecture_freeze_contract.yaml"
    )
    p_disposition = (
        BASELINE_AUDIT / "longrag_full_track_component_disposition.yaml"
    )
    p_envelope = (
        BASELINE_AUDIT / "longrag_full_track_dataset_envelope.yaml"
    )
    p_plan = BASELINE_AUDIT / "longrag_full_track_plan.yaml"

    write_yaml_atomic(p_embedding, embedding_profile)
    write_yaml_atomic(p_freeze, architecture_freeze)
    write_yaml_atomic(p_disposition, disposition)
    write_yaml_atomic(p_envelope, envelope)
    write_yaml_atomic(p_plan, plan)

    print("[6/8] writing verification report ...")
    verification = build_verification_report(
        sha_results, loader_results, registry
    )
    verification_path = OUTPUT_ROOT / "verification_report.json"
    write_json_atomic(verification_path, verification)

    print("[7/8] computing checksums ...")
    checksum_inputs = {
        (
            "baseline_audit/"
            "longrag_full_track_embedding_profile.yaml"
        ): p_embedding,
        (
            "baseline_audit/"
            "longrag_full_track_architecture_freeze_contract.yaml"
        ): p_freeze,
        (
            "baseline_audit/"
            "longrag_full_track_component_disposition.yaml"
        ): p_disposition,
        (
            "baseline_audit/"
            "longrag_full_track_dataset_envelope.yaml"
        ): p_envelope,
        "baseline_audit/longrag_full_track_plan.yaml": p_plan,
        (
            "outputs/external_longrag/_freeze-v1/"
            "verification_report.json"
        ): verification_path,
    }
    checksums = compute_checksums(checksum_inputs)

    print("[8/8] writing manifest ...")
    wall = time.perf_counter() - t0
    manifest: dict[str, Any] = {
        "artifact_type": (
            "longrag_full_track_architecture_freeze_manifest"
        ),
        "artifact_version": "1.0",
        "execution_id": "longrag-l1b-architecture-freeze-v3",
        "status": "SUCCEEDED",
        "wall_clock_seconds": round(wall, 3),
        "dataset_count": len(EXPECTED_DATASETS),
        "datasets": list(EXPECTED_DATASETS),
        "checksums": checksums,
        "outputs": sorted(checksum_inputs.keys()),
        "next_gate": "longrag_full_track_unit_materialization",
    }
    write_json_atomic(manifest_path, manifest)
    write_json_atomic(
        OUTPUT_ROOT / "checksums.json",
        {"files": checksums},
    )

    print()
    print(f"L1 COMPLETE in {wall:.2f}s")
    print(f"  outputs: {OUTPUT_ROOT}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore resume cache and re-run.",
    )
    parser.add_argument(
        "--skip-streaming",
        action="store_true",
        help="Skip the streaming loader verification pass.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return run(force=args.force, skip_streaming=args.skip_streaming)


if __name__ == "__main__":
    raise SystemExit(main())
