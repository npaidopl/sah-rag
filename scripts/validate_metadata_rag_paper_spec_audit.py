"""Validate the Metadata-RAG paper-to-spec audit."""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

AUDIT_PATH = Path("baseline_audit/metadata_rag_paper_spec_audit.yaml")

EXPECTED_ARCHITECTURE_ID = "metadata_rag"
EXPECTED_WAVE = 3
EXPECTED_PAPER_ID = "2510.24402"

ALLOWED_TERMINAL_STATUSES = {
    "CONDITIONAL_GO",
    "BLOCKED_WITH_REASON",
    "DEFERRED_WITH_REASON",
}
ALLOWED_ARCHITECTURE_CLASSES = {
    "ARCHITECTURAL_COMPARATOR",
    "PROJECT_OWNED_BASELINE",
}
ALLOWED_IMPLEMENTATION_CLASSES = {
    "PAPER_DERIVED_REIMPLEMENTATION",
    "PROJECT_OWNED_IMPLEMENTATION",
}
ALLOWED_BOUNDARY_STATUSES = {
    "PASS",
    "PARTIAL",
    "FAIL",
    "DESIGN_REQUIRED",
}
ALLOWED_TRACEABILITY_STATUSES = {
    "NATIVE_WITH_CANONICAL_MAPPING",
    "REQUIRES_INDEPENDENT_DESIGN",
    "NOT_RECOVERABLE",
    "UNRESOLVED",
}

LEAKAGE_KEYS = (
    "judgments_as_metadata",
    "reference_answers_as_metadata",
    "evaluation_labels_as_metadata",
    "generated_answers_as_metadata",
    "post_retrieval_annotations_as_metadata",
)


def main() -> int:
    if not AUDIT_PATH.is_file():
        raise SystemExit(f"missing audit: {AUDIT_PATH}")
    data = yaml.safe_load(AUDIT_PATH.read_text(encoding="utf-8"))

    arch = data["architecture"]
    if arch["id"] != EXPECTED_ARCHITECTURE_ID:
        raise SystemExit("architecture.id changed")
    if arch["wave"] != EXPECTED_WAVE:
        raise SystemExit("architecture.wave must be 3")

    identity = data["identity"]
    if identity["status"] not in ("RESOLVED", "UNRESOLVED"):
        raise SystemExit("identity.status invalid")
    if identity["paper_identifier"] != EXPECTED_PAPER_ID:
        raise SystemExit("paper_identifier changed")
    if not identity["paper_title"]:
        raise SystemExit("paper_title must be non-empty")

    fs = data["foreign_source"]
    if fs["repository_registered"] is not False:
        raise SystemExit("foreign_source.repository_registered must be false")
    if fs["checkout_present"] is not False:
        raise SystemExit("foreign_source.checkout_present must be false")
    if fs["foreign_source_license_applicable"] is not False:
        raise SystemExit("foreign_source.foreign_source_license_applicable must be false")
    if fs["foreign_source_execution_applicable"] is not False:
        raise SystemExit(
            "foreign_source.foreign_source_execution_applicable must be false"
        )

    algo = data["paper_algorithm"]
    for key in (
        "document_metadata_representation",
        "metadata_extraction",
        "metadata_normalization",
        "query_metadata_inference",
        "candidate_filtering",
        "semantic_retrieval",
        "metadata_semantic_fusion",
        "result_ranking",
        "source_lineage",
        "document_expansion",
    ):
        if algo[key] not in ("REQUIRED", "OPTIONAL", "OUT_OF_SCOPE", "UNRESOLVED"):
            raise SystemExit(f"paper_algorithm.{key} invalid")

    mc = data["metadata_contract"]
    if mc["schema_status"] not in (
        "RESOLVED",
        "REQUIRES_INDEPENDENT_DESIGN",
        "UNRESOLVED",
    ):
        raise SystemExit("metadata_contract.schema_status invalid")
    for key in (
        "field_name_normalization",
        "value_normalization",
        "missing_value_policy",
        "multi_value_policy",
        "unknown_field_policy",
    ):
        if key not in mc["normalization"]:
            raise SystemExit(f"metadata_contract.normalization missing {key}")

    qmi = data["query_metadata_inference"]
    if qmi["classification"] not in (
        "DETERMINISTIC_PROJECT_OWNED",
        "MODEL_ASSISTED_OPTIONAL",
        "MODEL_ASSISTED_REQUIRED",
        "UNRESOLVED",
    ):
        raise SystemExit("query_metadata_inference.classification invalid")

    rb = data["retrieval_boundary"]
    if rb["status"] not in ALLOWED_BOUNDARY_STATUSES:
        raise SystemExit("retrieval_boundary.status invalid")

    tr = data["traceability"]
    if tr["status"] not in ALLOWED_TRACEABILITY_STATUSES:
        raise SystemExit("traceability.status invalid")

    lp = data["leakage_policy"]
    for key in LEAKAGE_KEYS:
        if lp.get(key) != "prohibited":
            raise SystemExit(f"leakage_policy.{key} must be prohibited")

    rf = data["resource_feasibility"]
    for key in (
        "metadata_ingestion",
        "deterministic_normalization",
        "metadata_index",
        "dense_embedding",
        "query_metadata_inference",
        "hybrid_retrieval",
        "final_generator",
        "overall",
    ):
        if key not in rf:
            raise SystemExit(f"resource_feasibility missing {key}")

    cls = data["classification"]
    if cls["architecture_class"] not in ALLOWED_ARCHITECTURE_CLASSES:
        raise SystemExit("classification.architecture_class invalid")
    if cls["implementation_class"] not in ALLOWED_IMPLEMENTATION_CLASSES:
        raise SystemExit("classification.implementation_class invalid")

    # Paper-derived implementation requires a resolved paper identity.
    if (
        cls["implementation_class"] == "PAPER_DERIVED_REIMPLEMENTATION"
        and identity["status"] != "RESOLVED"
    ):
        raise SystemExit(
            "paper-derived implementation requires resolved paper identity"
        )

    disp = data["terminal_disposition"]
    if disp["status"] not in ALLOWED_TERMINAL_STATUSES:
        raise SystemExit(
            f"terminal_disposition.status must be one of {sorted(ALLOWED_TERMINAL_STATUSES)}"
        )
    if disp["status"] == "READY_FOR_DATA":
        raise SystemExit("READY_FOR_DATA is not permitted from an audit")
    if not disp.get("reason"):
        raise SystemExit("terminal_disposition.reason must not be empty")
    if not disp.get("next_action"):
        raise SystemExit("terminal_disposition.next_action must not be empty")

    if len(data["blocked_claims"]) < 5:
        raise SystemExit("blocked_claims must declare at least five claims")

    print("[OK] Metadata-RAG paper-to-spec audit")
    print(f"  identity status: {identity['status']}")
    print(f"  paper: {identity['paper_identifier']}")
    print(f"  paper venue: {identity['venue']}")
    print(f"  foreign repository registered: {fs['repository_registered']}")
    print(f"  foreign source execution applicable: {fs['foreign_source_execution_applicable']}")
    print(f"  metadata schema: {mc['schema_status']}")
    print(f"  query metadata inference: {qmi['classification']}")
    print(f"  retrieval fusion policy: {data['retrieval_fusion']['policy']}")
    print(f"  retrieval boundary: {rb['status']}")
    print(f"  traceability: {tr['status']}")
    print(f"  resource feasibility: {rf['overall']['status']}")
    print(f"  architecture class: {cls['architecture_class']}")
    print(f"  implementation class: {cls['implementation_class']}")
    print(f"  terminal disposition: {disp['status']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
