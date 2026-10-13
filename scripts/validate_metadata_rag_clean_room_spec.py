"""Validate the Metadata-RAG clean-room specification.

The validator is amendment-aware: it loads the frozen
``baseline_audit/metadata_rag_clean_room_spec.yaml`` plus every
``metadata_rag_clean_room_spec_amendment_*.yaml`` in ascending
numeric order, deep-merges their patches, and validates the resulting
effective spec. Any silent change to a frozen design decision must
fail this validator.

Running the validator without any amendment is an error, because the
project's own process requires every post-freeze decision to be
recorded as an explicit amendment.
"""

from __future__ import annotations

import math
import re
import sys
from pathlib import Path
from typing import Any, cast

import yaml

SPEC_PATH = Path("baseline_audit/metadata_rag_clean_room_spec.yaml")
AMENDMENTS_DIR = Path("baseline_audit")

EXPECTED_ARCHITECTURE_ID = "metadata_rag"
EXPECTED_ARXIV_ID = "2510.24402"
EXPECTED_FIDELITY_CLASS = "PAPER_DERIVED_COMPARATOR"
EXPECTED_IMPLEMENTATION_CLASS = "PAPER_DERIVED_REIMPLEMENTATION"
EXPECTED_SCHEMA_VERSION = "0.1.0"
EXPECTED_SCHEMA_STATUS = "RESOLVED"
EXPECTED_MODEL_ID = "BAAI/bge-m3"
EXPECTED_MODEL_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
EXPECTED_DENSE_DIMENSION = 1024
EXPECTED_ALPHA_METADATA = 0.3
EXPECTED_ALPHA_SEMANTIC = 0.7
EXPECTED_MIN_MATCH_RATIO = 0.5
EXPECTED_STORAGE_BACKEND = "IN_MEMORY_EXACT_DENSE"
EXPECTED_STORAGE_REPLACEMENT_CLASS = (
    "SEMANTICALLY_NEUTRAL_FOR_CONTROLLED_TRACK"
)
EXPECTED_EQUIVALENCE_TEST = "NOT_REQUIRED"
EXPECTED_CUTOFFS = (1, 5, 10, 20, 50, 100)
EXPECTED_PRIMARY_METRIC = "macro_document_recall_at_10"
EXPECTED_ZERO_RELEVANCE_POLICY = "exclude_with_explicit_count"
EXPECTED_AMENDMENT_IDS = (
    "metadata_rag_clean_room_spec_amendment_01",
    "metadata_rag_clean_room_spec_amendment_02",
    "metadata_rag_clean_room_spec_amendment_03",
    "metadata_rag_clean_room_spec_amendment_04",
)

FLOAT_TOLERANCE = 1e-9

LEAKAGE_KEYS: tuple[str, ...] = (
    "judgments_as_metadata",
    "reference_answers_as_metadata",
    "evaluation_labels_as_metadata",
    "generated_answers_as_metadata",
    "post_retrieval_annotations_as_metadata",
)

LEAKAGE_FALSE_KEYS: tuple[str, ...] = (
    "query_target_content_group_id_used",
    "query_doc_ids_used",
    "ground_truth_used_during_indexing",
    "ground_truth_used_during_querying",
)

REQUIRED_QUERY_PROHIBITED_FIELDS: frozenset[str] = frozenset(
    {
        "doc_ids",
        "relevant_ids",
        "relevant_doc_ids",
        "answers",
        "choice",
        "target_content_group_id",
        "ground_truth",
        "spans",
        "metadata",
    }
)

REQUIRED_QUERY_ALLOWED_FIELDS: frozenset[str] = frozenset(
    {"query_id", "query_text"}
)

REQUIRED_DATASETS: frozenset[str] = frozenset(
    {
        "ruler",
        "loong",
        "long_bench_v2",
        "lara",
        "greek_bar_bench",
        "cuad",
        "clerk",
    }
)

REQUIRED_VALUE_TYPES: frozenset[str] = frozenset(
    {
        "categorical",
        "temporal",
        "numeric",
        "entity",
        "structural",
        "provenance",
    }
)

REQUIRED_NORMALIZATION_KEYS: tuple[str, ...] = (
    "field_name_normalization",
    "value_normalization",
    "missing_value_policy",
    "multi_value_policy",
    "unknown_field_policy",
)

REQUIRED_ARCHITECTURE_DEFINING: frozenset[str] = frozenset(
    {
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
)

CLAIMS_FALSE: tuple[str, ...] = (
    "faithful_reproduction",
    "paper_effectiveness_reproduced",
    "native_equivalence",
    "real_dataset_validated",
    "publishable_result",
    "synthetic_validation_is_paper_level_evidence",
)

CLAIMS_TRUE: tuple[str, ...] = (
    "controlled_adaptation",
    "paper_derived_comparator",
)

_AMENDMENT_FILE_RE = re.compile(
    r"^metadata_rag_clean_room_spec_amendment_(\d+)\.yaml$"
)


def _amendment_paths() -> list[Path]:
    found: list[tuple[int, Path]] = []
    for path in AMENDMENTS_DIR.glob(
        "metadata_rag_clean_room_spec_amendment_*.yaml"
    ):
        match = _AMENDMENT_FILE_RE.match(path.name)
        if match is None:
            raise SystemExit(
                f"unexpected amendment filename: {path.name}"
            )
        found.append((int(match.group(1)), path))
    found.sort(key=lambda pair: pair[0])
    return [path for _, path in found]


def _deep_merge(target: dict[str, Any], patch: dict[str, Any]) -> None:
    for key, value in patch.items():
        existing = target.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            _deep_merge(
                cast("dict[str, Any]", existing),
                cast("dict[str, Any]", value),
            )
        else:
            target[key] = value


def _load_effective_spec() -> tuple[dict[str, Any], list[str]]:
    if not SPEC_PATH.is_file():
        raise SystemExit(f"missing spec: {SPEC_PATH}")
    raw = yaml.safe_load(SPEC_PATH.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise SystemExit("spec must parse to a mapping")
    effective: dict[str, Any] = cast("dict[str, Any]", raw)

    amendment_ids: list[str] = []
    for path in _amendment_paths():
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise SystemExit(f"{path.name}: amendment must be a mapping")
        am = payload.get("amendment")
        if not isinstance(am, dict):
            raise SystemExit(f"{path.name}: missing amendment block")
        am_id = am.get("id")
        if not isinstance(am_id, str) or not am_id:
            raise SystemExit(f"{path.name}: amendment.id missing")
        patch = am.get("patch")
        if not isinstance(patch, dict):
            raise SystemExit(f"{path.name}: amendment.patch missing")
        _deep_merge(effective, cast("dict[str, Any]", patch))
        amendment_ids.append(am_id)
    return effective, amendment_ids


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def _check_amendments(observed_ids: list[str]) -> None:
    if tuple(observed_ids) != EXPECTED_AMENDMENT_IDS:
        raise SystemExit(
            f"amendment set mismatch: "
            f"expected {EXPECTED_AMENDMENT_IDS}, got {tuple(observed_ids)}"
        )
    for path in _amendment_paths():
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        applies = data["amendment"]["applies_to"]
        declared = applies["sha256"]
        if not isinstance(declared, str) or not declared:
            raise SystemExit(f"{path.name}: applies_to.sha256 missing")
        if applies["path"] != str(SPEC_PATH).replace("\\", "/"):
            raise SystemExit(
                f"{path.name}: applies_to.path must point to the "
                "frozen spec"
            )


def _check_identity(data: dict[str, Any]) -> None:
    arch = data["architecture"]
    _require(
        arch["architecture_id"] == EXPECTED_ARCHITECTURE_ID,
        "architecture.architecture_id changed",
    )
    _require(
        arch["fidelity_class"] == EXPECTED_FIDELITY_CLASS,
        f"architecture.fidelity_class must be {EXPECTED_FIDELITY_CLASS}",
    )
    _require(
        arch["paper"]["arxiv_id"] == EXPECTED_ARXIV_ID,
        "architecture.paper.arxiv_id changed",
    )
    _require(
        arch["classification"]["implementation_class"]
        == EXPECTED_IMPLEMENTATION_CLASS,
        "classification.implementation_class changed",
    )


def _check_foreign_source(data: dict[str, Any]) -> None:
    fs = data["foreign_source"]
    for key in (
        "repository_registered",
        "checkout_present",
        "foreign_source_license_applicable",
        "foreign_source_execution_applicable",
        "foreign_source_reuse_applicable",
    ):
        _require(fs[key] is False, f"foreign_source.{key} must be false")
    _require(
        fs["license_status"] == "NOT_APPLICABLE",
        "foreign_source.license_status must be NOT_APPLICABLE",
    )


def _check_tracks(data: dict[str, Any]) -> None:
    tracks = data["tracks"]
    _require(
        tracks["controlled"]["eligible"] is True,
        "tracks.controlled.eligible must be true",
    )
    _require(
        tracks["native"]["eligible"] is False,
        "tracks.native.eligible must be false",
    )
    _require(
        tracks["native"]["reason"] == "DEFERRED_WITH_REASON",
        "tracks.native.reason must be DEFERRED_WITH_REASON",
    )


def _check_metadata_schema(data: dict[str, Any]) -> None:
    ms = data["metadata_schema"]
    _require(
        ms["schema_version"] == EXPECTED_SCHEMA_VERSION,
        "metadata_schema.schema_version changed",
    )
    _require(
        ms["schema_status"] == EXPECTED_SCHEMA_STATUS,
        "metadata_schema.schema_status must be RESOLVED",
    )
    declared = set(ms["value_types_declared"])
    missing = REQUIRED_VALUE_TYPES - declared
    _require(
        not missing,
        f"metadata_schema.value_types_declared missing: {sorted(missing)}",
    )
    norm = ms["normalization"]
    for key in REQUIRED_NORMALIZATION_KEYS:
        _require(key in norm, f"metadata_schema.normalization missing {key}")


def _check_query_projection(data: dict[str, Any]) -> None:
    qmi = data["query_metadata_inference"]
    _require(
        qmi["classification"] == "DETERMINISTIC_PROJECT_OWNED",
        "query_metadata_inference.classification changed",
    )
    _require(
        qmi["llm_required"] is False,
        "query_metadata_inference.llm_required must be false",
    )
    _require(
        qmi["hosted_model_required"] is False,
        "query_metadata_inference.hosted_model_required must be false",
    )
    proj = qmi["query_projection"]
    _require(
        set(proj["allowed_fields"]) == REQUIRED_QUERY_ALLOWED_FIELDS,
        "query_projection.allowed_fields must be exactly "
        "{query_id, query_text}",
    )
    prohibited = set(proj["prohibited_fields"])
    missing = REQUIRED_QUERY_PROHIBITED_FIELDS - prohibited
    _require(
        not missing,
        f"query_projection.prohibited_fields missing: {sorted(missing)}",
    )


def _check_leakage(data: dict[str, Any]) -> None:
    lp = data["leakage_policy"]
    for key in LEAKAGE_KEYS:
        _require(
            lp.get(key) == "prohibited",
            f"leakage_policy.{key} must be prohibited",
        )
    for key in LEAKAGE_FALSE_KEYS:
        _require(
            lp.get(key) is False,
            f"leakage_policy.{key} must be false",
        )


def _check_fusion(data: dict[str, Any]) -> None:
    cf = data["candidate_filtering"]
    _require(
        cf["min_match_ratio_frozen"] is True,
        "candidate_filtering.min_match_ratio_frozen must be true",
    )
    _require(
        cf["min_match_ratio_tuning_prohibited"] is True,
        "candidate_filtering.min_match_ratio_tuning_prohibited "
        "must be true",
    )
    _require(
        math.isclose(
            float(cf["min_match_ratio"]),
            EXPECTED_MIN_MATCH_RATIO,
            abs_tol=FLOAT_TOLERANCE,
        ),
        "candidate_filtering.min_match_ratio changed",
    )

    fusion = data["metadata_semantic_fusion"]
    coef = fusion["coefficients"]
    _require(
        coef["frozen"] is True,
        "fusion.coefficients.frozen must be true",
    )
    _require(
        coef["tuning_prohibited"] is True,
        "fusion.coefficients.tuning_prohibited must be true",
    )
    _require(
        math.isclose(
            float(coef["alpha_metadata"]),
            EXPECTED_ALPHA_METADATA,
            abs_tol=FLOAT_TOLERANCE,
        ),
        "fusion.coefficients.alpha_metadata changed",
    )
    _require(
        math.isclose(
            float(coef["alpha_semantic"]),
            EXPECTED_ALPHA_SEMANTIC,
            abs_tol=FLOAT_TOLERANCE,
        ),
        "fusion.coefficients.alpha_semantic changed",
    )
    total = float(coef["alpha_metadata"]) + float(coef["alpha_semantic"])
    _require(
        math.isclose(total, 1.0, abs_tol=FLOAT_TOLERANCE),
        f"fusion coefficients must sum to 1.0, got {total}",
    )


def _check_embedder(data: dict[str, Any]) -> None:
    sr = data["semantic_retrieval"]
    _require(
        sr["signal"] == "dense_only",
        "semantic_retrieval.signal must be dense_only",
    )
    _require(
        sr["sparse_enabled"] is False,
        "semantic_retrieval.sparse_enabled must be false",
    )
    _require(
        sr["reranking_enabled"] is False,
        "semantic_retrieval.reranking_enabled must be false",
    )
    _require(
        sr["hybrid_enabled"] is False,
        "semantic_retrieval.hybrid_enabled must be false",
    )
    _require(
        sr["similarity"] == "COSINE",
        "semantic_retrieval.similarity must be COSINE",
    )
    emb = sr["embedder"]
    _require(
        emb["treatment"] == "CONTROLLED_REPLACEMENT",
        "embedder.treatment must be CONTROLLED_REPLACEMENT",
    )
    _require(
        emb["model_id"] == EXPECTED_MODEL_ID,
        "embedder.model_id changed",
    )
    _require(
        emb["model_revision"] == EXPECTED_MODEL_REVISION,
        "embedder.model_revision changed",
    )
    _require(
        int(emb["dense_dimension"]) == EXPECTED_DENSE_DIMENSION,
        "embedder.dense_dimension changed",
    )
    _require(
        emb["dense_normalized"] is True,
        "embedder.dense_normalized must be true",
    )
    _require(
        emb["network_fallback_allowed"] is False,
        "embedder.network_fallback_allowed must be false",
    )


def _check_storage(data: dict[str, Any]) -> None:
    st = data["storage"]
    _require(
        st["experimental_backend"] == EXPECTED_STORAGE_BACKEND,
        f"storage.experimental_backend must be {EXPECTED_STORAGE_BACKEND}",
    )
    _require(
        st["replacement_class"] == EXPECTED_STORAGE_REPLACEMENT_CLASS,
        "storage.replacement_class must be "
        f"{EXPECTED_STORAGE_REPLACEMENT_CLASS}",
    )
    _require(
        st["equivalence_test"] == EXPECTED_EQUIVALENCE_TEST,
        f"storage.equivalence_test must be {EXPECTED_EQUIVALENCE_TEST}",
    )
    _require(
        st["approximate_search_allowed"] is False,
        "storage.approximate_search_allowed must be false",
    )
    _require(
        st["quantization_allowed"] is False,
        "storage.quantization_allowed must be false",
    )
    _require(
        st["vector_name"] == "dense",
        "storage.vector_name must be dense",
    )
    _require(
        st["distance"] == "COSINE",
        "storage.distance must be COSINE",
    )
    _require(
        int(st["dimension"]) == EXPECTED_DENSE_DIMENSION,
        "storage.dimension changed",
    )
    _require(
        st.get("reference_index_module")
        == "rag_system.baselines.metadata_rag_reference_index",
        "storage.reference_index_module must point to the "
        "project-owned reference index module",
    )


def _check_datasets(data: dict[str, Any]) -> None:
    dc = data["dataset_compatibility"]
    ds = set(dc["applicable_datasets"])
    missing = REQUIRED_DATASETS - ds
    _require(
        not missing,
        f"dataset_compatibility.applicable_datasets missing: "
        f"{sorted(missing)}",
    )
    _require(
        dc["evaluation_level"] == "canonical_document",
        "dataset_compatibility.evaluation_level must be "
        "canonical_document",
    )
    observed_cutoffs = tuple(int(c) for c in dc["cutoffs"])
    _require(
        observed_cutoffs == EXPECTED_CUTOFFS,
        f"dataset_compatibility.cutoffs must be {EXPECTED_CUTOFFS}, "
        f"got {observed_cutoffs}",
    )
    _require(
        dc.get("primary_metric") == EXPECTED_PRIMARY_METRIC,
        "dataset_compatibility.primary_metric must be "
        f"{EXPECTED_PRIMARY_METRIC}",
    )
    _require(
        dc.get("zero_relevance_policy") == EXPECTED_ZERO_RELEVANCE_POLICY,
        "dataset_compatibility.zero_relevance_policy must be "
        f"{EXPECTED_ZERO_RELEVANCE_POLICY}",
    )


def _check_windowing(data: dict[str, Any]) -> None:
    ms = data["metadata_schema"]
    windowing = ms.get("windowing")
    _require(
        isinstance(windowing, dict),
        "metadata_schema.windowing must be present (amendment_04)",
    )
    _require(
        windowing["window_content_tokens"] == 8190,
        "windowing.window_content_tokens must be 8190",
    )
    _require(
        windowing["maximum_model_input_tokens"] == 8192,
        "windowing.maximum_model_input_tokens must be 8192",
    )
    _require(
        windowing["special_token_count"] == 2,
        "windowing.special_token_count must be 2",
    )
    _require(
        windowing["tokenizer_id"] == EXPECTED_MODEL_ID,
        "windowing.tokenizer_id must equal the frozen embedder",
    )
    _require(
        windowing["tokenizer_revision"] == EXPECTED_MODEL_REVISION,
        "windowing.tokenizer_revision must equal the frozen embedder",
    )
    _require(
        windowing["lineage_field"] == "lineage_doc_id",
        "windowing.lineage_field must be lineage_doc_id",
    )
    boundary = data["retrieval_boundary"]
    _require(
        boundary.get("architecture_item_id_semantics") == "window_id",
        "retrieval_boundary.architecture_item_id_semantics must be "
        "window_id",
    )
    _require(
        boundary.get("canonical_doc_id_semantics") == "lineage_doc_id",
        "retrieval_boundary.canonical_doc_id_semantics must be "
        "lineage_doc_id",
    )
    expansion = data["document_expansion"]
    _require(
        expansion.get("deduplication_key") == "canonical_doc_id",
        "document_expansion.deduplication_key must be canonical_doc_id",
    )


def _check_fidelity_boundary(data: dict[str, Any]) -> None:
    fb = data["fidelity_boundary"]
    defining = set(fb["architecture_defining"])
    missing = REQUIRED_ARCHITECTURE_DEFINING - defining
    _require(
        not missing,
        f"fidelity_boundary.architecture_defining missing: {sorted(missing)}",
    )


def _check_claims(data: dict[str, Any]) -> None:
    cl = data["claims"]
    for key in CLAIMS_FALSE:
        _require(cl[key] is False, f"claims.{key} must be false")
    for key in CLAIMS_TRUE:
        _require(cl[key] is True, f"claims.{key} must be true")


def main() -> int:
    effective, amendment_ids = _load_effective_spec()
    _check_amendments(amendment_ids)
    _check_identity(effective)
    _check_foreign_source(effective)
    _check_tracks(effective)
    _check_metadata_schema(effective)
    _check_query_projection(effective)
    _check_leakage(effective)
    _check_fusion(effective)
    _check_embedder(effective)
    _check_storage(effective)
    _check_datasets(effective)
    _check_fidelity_boundary(effective)
    _check_windowing(effective)
    _check_claims(effective)

    dc = effective["dataset_compatibility"]
    print("[OK] Metadata-RAG clean-room spec (effective)")
    print(f"  amendments_applied = {amendment_ids}")
    print(
        "  fidelity_class     = "
        f"{effective['architecture']['fidelity_class']}"
    )
    print(
        "  schema_version     = "
        f"{effective['metadata_schema']['schema_version']}"
    )
    coef = effective["metadata_semantic_fusion"]["coefficients"]
    print(f"  alpha_metadata     = {coef['alpha_metadata']}")
    print(f"  alpha_semantic     = {coef['alpha_semantic']}")
    print(
        "  min_match_ratio    = "
        f"{effective['candidate_filtering']['min_match_ratio']}"
    )
    emb = effective["semantic_retrieval"]["embedder"]
    print(f"  model_id           = {emb['model_id']}")
    print(f"  model_revision     = {emb['model_revision']}")
    print(
        "  storage_backend    = "
        f"{effective['storage']['experimental_backend']}"
    )
    print(
        "  storage_equivalence= "
        f"{effective['storage']['equivalence_test']}"
    )
    print(f"  cutoffs            = {dc['cutoffs']}")
    print(f"  primary_metric     = {dc['primary_metric']}")
    print(f"  zero_relevance     = {dc['zero_relevance_policy']}")
    print(
        "  controlled_eligible= "
        f"{effective['tracks']['controlled']['eligible']}"
    )
    print(
        "  native_eligible    = "
        f"{effective['tracks']['native']['eligible']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
