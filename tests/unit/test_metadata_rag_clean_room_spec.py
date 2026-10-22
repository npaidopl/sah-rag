"""Tests for the Metadata-RAG clean-room specification."""

from __future__ import annotations

import hashlib
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC = REPO_ROOT / "baseline_audit" / "metadata_rag_clean_room_spec.yaml"
AMENDMENT_01 = (
    REPO_ROOT
    / "baseline_audit"
    / "metadata_rag_clean_room_spec_amendment_01.yaml"
)
AMENDMENT_02 = (
    REPO_ROOT
    / "baseline_audit"
    / "metadata_rag_clean_room_spec_amendment_02.yaml"
)
AMENDMENT_03 = (
    REPO_ROOT
    / "baseline_audit"
    / "metadata_rag_clean_room_spec_amendment_03.yaml"
)
AMENDMENT_04 = (
    REPO_ROOT
    / "baseline_audit"
    / "metadata_rag_clean_room_spec_amendment_04.yaml"
)
VALIDATOR = (
    REPO_ROOT / "scripts" / "validate_metadata_rag_clean_room_spec.py"
)

FROZEN_SPEC_SHA256 = (
    "A02DBA61EE1DD922F3B2DE0CED932E7E20E748C381E8AACC655A070BEC50DB02"
)

ALL_AMENDMENTS = (AMENDMENT_01, AMENDMENT_02, AMENDMENT_03, AMENDMENT_04)


def _load() -> dict[str, Any]:
    raw = yaml.safe_load(SPEC.read_text(encoding="utf-8"))
    assert isinstance(raw, dict)
    return raw


def _deep_merge(target: dict[str, Any], patch: dict[str, Any]) -> None:
    for key, value in patch.items():
        existing = target.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            _deep_merge(existing, value)
        else:
            target[key] = value


def _load_effective() -> dict[str, Any]:
    effective = _load()
    for path in ALL_AMENDMENTS:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        _deep_merge(effective, payload["amendment"]["patch"])
    return effective


def _run_validator() -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(VALIDATOR)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )


# --- Spec identity ---------------------------------------------------------


def test_spec_file_exists() -> None:
    assert SPEC.is_file(), f"missing spec: {SPEC}"


def test_frozen_spec_sha256_unchanged() -> None:
    observed = hashlib.sha256(SPEC.read_bytes()).hexdigest().upper()
    assert observed == FROZEN_SPEC_SHA256, (
        "the frozen spec file must not be edited; use an amendment instead"
    )


def test_all_amendments_present() -> None:
    for path in ALL_AMENDMENTS:
        assert path.is_file(), f"missing amendment: {path.name}"


def test_amendments_bind_to_frozen_spec_sha256() -> None:
    for path in ALL_AMENDMENTS:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        declared = payload["amendment"]["applies_to"]["sha256"]
        assert declared == FROZEN_SPEC_SHA256, path.name


# --- Validator -------------------------------------------------------------


def test_validator_accepts_effective_spec() -> None:
    result = _run_validator()
    assert result.returncode == 0, result.stderr
    assert (
        "[OK] Metadata-RAG clean-room spec (effective)" in result.stdout
    )
    assert (
        "metadata_rag_clean_room_spec_amendment_01" in result.stdout
    )
    assert (
        "metadata_rag_clean_room_spec_amendment_02" in result.stdout
    )
    assert (
        "metadata_rag_clean_room_spec_amendment_03" in result.stdout
    )
    assert (
        "metadata_rag_clean_room_spec_amendment_04" in result.stdout
    )


# --- Effective spec --------------------------------------------------------


def test_effective_identity_is_paper_derived_comparator() -> None:
    data = _load_effective()
    arch = data["architecture"]
    assert arch["architecture_id"] == "metadata_rag"
    assert arch["fidelity_class"] == "PAPER_DERIVED_COMPARATOR"
    assert (
        arch["classification"]["implementation_class"]
        == "PAPER_DERIVED_REIMPLEMENTATION"
    )
    assert arch["paper"]["arxiv_id"] == "2510.24402"


def test_effective_no_foreign_source() -> None:
    data = _load_effective()
    fs = data["foreign_source"]
    assert fs["repository_registered"] is False
    assert fs["checkout_present"] is False
    assert fs["foreign_source_execution_applicable"] is False
    assert fs["license_status"] == "NOT_APPLICABLE"


def test_effective_controlled_track_eligible_native_deferred() -> None:
    data = _load_effective()
    assert data["tracks"]["controlled"]["eligible"] is True
    assert data["tracks"]["native"]["eligible"] is False
    assert data["tracks"]["native"]["reason"] == "DEFERRED_WITH_REASON"


def test_effective_metadata_schema_resolved() -> None:
    data = _load_effective()
    ms = data["metadata_schema"]
    assert ms["schema_version"] == "0.1.0"
    assert ms["schema_status"] == "RESOLVED"


def test_effective_normalization_policies_present() -> None:
    data = _load_effective()
    norm = data["metadata_schema"]["normalization"]
    for key in (
        "field_name_normalization",
        "value_normalization",
        "missing_value_policy",
        "multi_value_policy",
        "unknown_field_policy",
    ):
        assert key in norm, key
        assert norm[key], f"{key} must be non-empty"


def test_effective_query_metadata_inference_is_deterministic() -> None:
    data = _load_effective()
    qmi = data["query_metadata_inference"]
    assert qmi["classification"] == "DETERMINISTIC_PROJECT_OWNED"
    assert qmi["llm_required"] is False
    assert qmi["hosted_model_required"] is False


def test_effective_query_projection_allowed_fields_exact() -> None:
    data = _load_effective()
    allowed = set(
        data["query_metadata_inference"]["query_projection"]["allowed_fields"]
    )
    assert allowed == {"query_id", "query_text"}


def test_effective_query_projection_prohibited_fields_complete() -> None:
    data = _load_effective()
    prohibited = set(
        data["query_metadata_inference"]["query_projection"][
            "prohibited_fields"
        ]
    )
    for key in (
        "doc_ids",
        "relevant_ids",
        "relevant_doc_ids",
        "answers",
        "choice",
        "target_content_group_id",
        "ground_truth",
        "spans",
        "metadata",
    ):
        assert key in prohibited, key


def test_effective_leakage_policy_prohibits_all_five() -> None:
    data = _load_effective()
    lp = data["leakage_policy"]
    for key in (
        "judgments_as_metadata",
        "reference_answers_as_metadata",
        "evaluation_labels_as_metadata",
        "generated_answers_as_metadata",
        "post_retrieval_annotations_as_metadata",
    ):
        assert lp[key] == "prohibited", key


def test_effective_query_target_id_and_doc_ids_not_used() -> None:
    data = _load_effective()
    lp = data["leakage_policy"]
    assert lp["query_target_content_group_id_used"] is False
    assert lp["query_doc_ids_used"] is False
    assert lp["ground_truth_used_during_indexing"] is False
    assert lp["ground_truth_used_during_querying"] is False


def test_effective_fusion_coefficients_frozen_and_sum_to_one() -> None:
    data = _load_effective()
    coef = data["metadata_semantic_fusion"]["coefficients"]
    assert coef["frozen"] is True
    assert coef["tuning_prohibited"] is True
    assert math.isclose(float(coef["alpha_metadata"]), 0.3, abs_tol=1e-9)
    assert math.isclose(float(coef["alpha_semantic"]), 0.7, abs_tol=1e-9)
    total = float(coef["alpha_metadata"]) + float(coef["alpha_semantic"])
    assert math.isclose(total, 1.0, abs_tol=1e-9)


def test_effective_min_match_ratio_frozen() -> None:
    data = _load_effective()
    cf = data["candidate_filtering"]
    assert cf["min_match_ratio_frozen"] is True
    assert cf["min_match_ratio_tuning_prohibited"] is True
    assert math.isclose(float(cf["min_match_ratio"]), 0.5, abs_tol=1e-9)


def test_effective_embedder_pinned_bge_m3_dense_only() -> None:
    data = _load_effective()
    sr = data["semantic_retrieval"]
    assert sr["signal"] == "dense_only"
    assert sr["sparse_enabled"] is False
    assert sr["hybrid_enabled"] is False
    assert sr["reranking_enabled"] is False
    emb = sr["embedder"]
    assert emb["model_id"] == "BAAI/bge-m3"
    assert (
        emb["model_revision"]
        == "5617a9f61b028005a4858fdac845db406aefb181"
    )
    assert emb["dense_dimension"] == 1024
    assert emb["dense_normalized"] is True
    assert emb["network_fallback_allowed"] is False


def test_effective_storage_is_in_memory_exact_dense() -> None:
    data = _load_effective()
    st = data["storage"]
    assert st["experimental_backend"] == "IN_MEMORY_EXACT_DENSE"
    assert (
        st["replacement_class"]
        == "SEMANTICALLY_NEUTRAL_FOR_CONTROLLED_TRACK"
    )
    assert st["equivalence_test"] == "NOT_REQUIRED"
    assert st["approximate_search_allowed"] is False
    assert st["quantization_allowed"] is False
    assert (
        st["reference_index_module"]
        == "baselines.metadata_rag_reference_index"
    )


def test_effective_all_seven_datasets_declared() -> None:
    data = _load_effective()
    declared = set(data["dataset_compatibility"]["applicable_datasets"])
    expected = {
        "ruler",
        "loong",
        "long_bench_v2",
        "lara",
        "greek_bar_bench",
        "cuad",
        "clerk",
    }
    assert declared == expected


def test_effective_cutoffs_aligned_with_executed_runs() -> None:
    data = _load_effective()
    dc = data["dataset_compatibility"]
    assert tuple(dc["cutoffs"]) == (1, 5, 10, 20, 50, 100)
    assert dc["primary_metric"] == "macro_document_recall_at_10"
    assert dc["zero_relevance_policy"] == "exclude_with_explicit_count"


def test_effective_architecture_defining_components_complete() -> None:
    data = _load_effective()
    defining = set(data["fidelity_boundary"]["architecture_defining"])
    expected = {
        "metadata_schema",
        "metadata_extraction",
        "query_metadata_inference",
        "candidate_filtering",
        "semantic_retrieval",
        "metadata_semantic_fusion",
        "result_ranking",
        "source_lineage",
        "document_expansion",
    }
    assert defining == expected


def test_effective_claims_conservative() -> None:
    data = _load_effective()
    cl = data["claims"]
    assert cl["faithful_reproduction"] is False
    assert cl["paper_effectiveness_reproduced"] is False
    assert cl["native_equivalence"] is False
    assert cl["controlled_adaptation"] is True
    assert cl["paper_derived_comparator"] is True
    assert cl["real_dataset_validated"] is False
    assert cl["publishable_result"] is False
    assert cl["synthetic_validation_is_paper_level_evidence"] is False


def test_effective_capitalized_phrase_pattern_excludes_article() -> None:
    data = _load_effective()
    pattern = data["metadata_schema"]["derived_tier"]["frozen_patterns"][
        "capitalized_phrase"
    ]
    assert "(?!" in pattern, (
        "the effective capitalized_phrase pattern must carry the "
        "amendment_01 negative-lookahead guard"
    )


def test_effective_windowing_is_amendment_04() -> None:
    data = _load_effective()
    windowing = data["metadata_schema"]["windowing"]
    assert windowing["window_content_tokens"] == 8190
    assert windowing["maximum_model_input_tokens"] == 8192
    assert windowing["special_token_count"] == 2
    assert windowing["tokenizer_id"] == "BAAI/bge-m3"
    assert (
        windowing["tokenizer_revision"]
        == "5617a9f61b028005a4858fdac845db406aefb181"
    )
    assert windowing["lineage_field"] == "lineage_doc_id"


def test_effective_retrieval_boundary_uses_lineage() -> None:
    data = _load_effective()
    boundary = data["retrieval_boundary"]
    assert boundary["architecture_item_id_semantics"] == "window_id"
    assert boundary["canonical_doc_id_semantics"] == "lineage_doc_id"


def test_effective_document_expansion_dedupes_by_canonical() -> None:
    data = _load_effective()
    expansion = data["document_expansion"]
    assert expansion["method"] == "first_occurrence_by_rank"
    assert expansion["deduplication_key"] == "canonical_doc_id"
