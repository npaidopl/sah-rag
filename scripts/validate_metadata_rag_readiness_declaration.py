"""Validate the Metadata-RAG readiness declaration.

The validator performs two independent checks:

  1. Structural: the declaration contains every required block, every
     required key, and every claim is set to the expected value.

  2. Integrity: every entry in ``artifact_registry`` refers to a file
     that exists on disk under the repository root, and the SHA-256 of
     that file matches the declared hash. Any mismatch is a hard
     failure.

Both checks are required. A structurally valid declaration with
drifted artifact hashes is not a valid declaration.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any, cast

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DECLARATION_PATH = (
    REPO_ROOT
    / "baseline_audit"
    / "metadata_rag_readiness_declaration.yaml"
)

EXPECTED_ARTIFACT_TYPE = "metadata_rag_readiness_declaration"
EXPECTED_ARCHITECTURE_ID = "metadata_rag"
EXPECTED_FIDELITY_CLASS = "PAPER_DERIVED_COMPARATOR"
EXPECTED_IMPLEMENTATION_CLASS = "PAPER_DERIVED_REIMPLEMENTATION"
EXPECTED_DECLARATION_STATUS = "REAL_DATASET_EXECUTED"

REQUIRED_AMENDMENTS: frozenset[str] = frozenset(
    {
        "baseline_audit/metadata_rag_clean_room_spec_amendment_01.yaml",
        "baseline_audit/metadata_rag_clean_room_spec_amendment_02.yaml",
        "baseline_audit/metadata_rag_clean_room_spec_amendment_03.yaml",
    }
)

REQUIRED_PRODUCTION_MODULES: frozenset[str] = frozenset(
    {
        "src/sah_rag/baselines/metadata_rag_metadata.py",
        "src/sah_rag/baselines/metadata_rag_adapter.py",
        "src/sah_rag/baselines/metadata_rag_state.py",
        "src/sah_rag/baselines/metadata_rag_persistence.py",
        "src/sah_rag/baselines/metadata_rag_embedder.py",
        "src/sah_rag/baselines/metadata_rag_reference_index.py",
        "src/sah_rag/baselines/metadata_rag_retrieval.py",
        "src/sah_rag/baselines/metadata_rag_expansion.py",
        "src/sah_rag/baselines/metadata_rag_evaluation_case.py",
    }
)

REQUIRED_TESTS: frozenset[str] = frozenset(
    {
        "tests/unit/test_metadata_rag_paper_spec_audit.py",
        "tests/unit/test_metadata_rag_metadata.py",
        "tests/unit/test_metadata_rag_adapter.py",
        "tests/unit/test_metadata_rag_persistence.py",
        "tests/unit/test_metadata_rag_clean_room_spec.py",
        "tests/unit/test_metadata_rag_embedder.py",
        "tests/unit/test_metadata_rag_reference_index.py",
        "tests/unit/test_metadata_rag_retrieval.py",
        "tests/unit/test_metadata_rag_expansion.py",
        "tests/unit/test_metadata_rag_evaluation_case.py",
        "tests/unit/test_metadata_rag_synthetic_acceptance.py",
    }
)

REQUIRED_GATES: tuple[str, ...] = (
    "canonical_input_accepted",
    "preprocessing_completed",
    "query_count_matches",
    "ranked_results_nonempty",
    "canonical_ids_recovered",
    "duplicate_ids_preserved",
    "every_query_top2_matches_ground_truth",
    "fallback_not_triggered_on_fixture",
    "evaluation_case_created",
    "generation_disabled",
    "dataset_id_consistent",
    "state_reopened",
    "repeat_run_deterministic_or_classified",
)

CLAIMS_TRUE: tuple[str, ...] = (
    "synthetic_validated",
    "architecture_defining_components_preserved",
    "controlled_adaptation",
    "paper_derived_comparator",
)

CLAIMS_FALSE: tuple[str, ...] = (
    "faithful_reproduction",
    "paper_effectiveness_reproduced",
    "native_equivalence",
    "publishable_result",
    "synthetic_validation_is_paper_level_evidence",
)

LEAKAGE_KEYS: tuple[str, ...] = (
    "judgments_as_metadata",
    "reference_answers_as_metadata",
    "evaluation_labels_as_metadata",
    "generated_answers_as_metadata",
    "post_retrieval_annotations_as_metadata",
)

REQUIRED_REAL_DATASETS: frozenset[str] = frozenset(
    {"clerk", "cuad", "greek_bar_bench", "lara"}
)

REQUIRED_CUTOFFS: tuple[int, ...] = (1, 5, 10, 20, 50, 100)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def _load() -> dict[str, Any]:
    if not DECLARATION_PATH.is_file():
        raise SystemExit(f"missing declaration: {DECLARATION_PATH}")
    raw = yaml.safe_load(DECLARATION_PATH.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise SystemExit("declaration must parse to a mapping")
    return cast("dict[str, Any]", raw)


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest().upper()


# -- Structural checks -----------------------------------------------------


def _check_top_level(data: dict[str, Any]) -> None:
    _require(
        data.get("artifact_type") == EXPECTED_ARTIFACT_TYPE,
        "artifact_type mismatch",
    )
    decl = data["declaration"]
    _require(
        decl["status"] == EXPECTED_DECLARATION_STATUS,
        "declaration.status must be SYNTHETIC_VALIDATED",
    )


def _check_architecture(data: dict[str, Any]) -> None:
    arch = data["architecture"]
    _require(
        arch["architecture_id"] == EXPECTED_ARCHITECTURE_ID,
        "architecture.architecture_id mismatch",
    )
    _require(
        arch["fidelity_class"] == EXPECTED_FIDELITY_CLASS,
        "architecture.fidelity_class mismatch",
    )
    _require(
        arch["implementation_class"] == EXPECTED_IMPLEMENTATION_CLASS,
        "architecture.implementation_class mismatch",
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


def _check_registry_paths(data: dict[str, Any]) -> None:
    registry = data["artifact_registry"]
    _require(
        isinstance(registry, list) and len(registry) > 0,
        "artifact_registry must be a non-empty list",
    )
    paths = {entry["path"] for entry in registry}
    missing_amendments = REQUIRED_AMENDMENTS - paths
    _require(
        not missing_amendments,
        f"artifact_registry missing amendments: "
        f"{sorted(missing_amendments)}",
    )
    missing_modules = REQUIRED_PRODUCTION_MODULES - paths
    _require(
        not missing_modules,
        f"artifact_registry missing production modules: "
        f"{sorted(missing_modules)}",
    )
    missing_tests = REQUIRED_TESTS - paths
    _require(
        not missing_tests,
        f"artifact_registry missing tests: {sorted(missing_tests)}",
    )


def _check_gates(data: dict[str, Any]) -> None:
    acceptance = data["synthetic_acceptance"]
    _require(
        acceptance["fixture"] == "infrastructure-pilot-synthetic-v1",
        "synthetic_acceptance.fixture mismatch",
    )
    _require(
        acceptance["document_count"] == 12,
        "synthetic_acceptance.document_count must be 12",
    )
    _require(
        acceptance["query_count"] == 6,
        "synthetic_acceptance.query_count must be 6",
    )
    _require(
        acceptance["encoder_kind"] == "DETERMINISTIC_CONTRACT_ENCODER",
        "synthetic_acceptance.encoder_kind mismatch",
    )
    gates = acceptance["gates"]
    for gate in REQUIRED_GATES:
        _require(gate in gates, f"gate missing: {gate}")
        _require(
            gates[gate] is True,
            f"gate must be true: {gate}",
        )


def _check_real_dataset_status(data: dict[str, Any]) -> None:
    rde = data["real_dataset_execution"]
    _require(
        rde["status"] == "EXECUTED",
        "real_dataset_execution.status must be EXECUTED",
    )
    scope = set(rde["executed_scope"])
    _require(
        scope == REQUIRED_REAL_DATASETS,
        "real_dataset_execution.executed_scope mismatch",
    )
    embedder = rde["embedder"]
    _require(
        embedder["model_id"] == "BAAI/bge-m3",
        "embedder.model_id mismatch",
    )
    _require(
        embedder["model_revision"]
        == "5617a9f61b028005a4858fdac845db406aefb181",
        "embedder.model_revision mismatch",
    )
    _require(
        embedder["mode"] == "dense_only",
        "embedder.mode must be dense_only",
    )
    protocol = rde["protocol"]
    _require(
        protocol["query_projection"] == ["query_id", "query_text"],
        "protocol.query_projection mismatch",
    )
    _require(
        protocol["generation_enabled"] is False,
        "protocol.generation_enabled must be false",
    )
    _require(
        protocol["llm_judge_enabled"] is False,
        "protocol.llm_judge_enabled must be false",
    )
    _require(
        tuple(protocol["cutoffs"]) == REQUIRED_CUTOFFS,
        "protocol.cutoffs mismatch",
    )
    _require(
        protocol["primary_metric"] == "macro_document_recall_at_10",
        "protocol.primary_metric mismatch",
    )


def _check_results(data: dict[str, Any]) -> None:
    results = data["results"]
    _require(
        results["primary_metric"] == "macro_document_recall_at_10",
        "results.primary_metric mismatch",
    )
    per_ds = results["per_dataset"]
    for ds in REQUIRED_REAL_DATASETS:
        _require(
            ds in per_ds,
            f"results.per_dataset missing dataset: {ds}",
        )
        value = float(per_ds[ds])
        _require(
            0.0 <= value <= 1.0,
            f"results.per_dataset.{ds} must be in [0, 1]",
        )
    macro = float(results["macro"])
    _require(
        0.0 <= macro <= 1.0,
        "results.macro must be in [0, 1]",
    )
    three = results["three_track_comparison"]
    _require(
        set(three.keys()) == {"longrag", "metadata_rag", "sah_rag"},
        "results.three_track_comparison must contain exactly "
        "longrag, metadata_rag, sah_rag",
    )


def _check_architectural_observation(data: dict[str, Any]) -> None:
    obs = data["architectural_observation"]
    rates = obs["fallback_rate_by_dataset"]
    for ds in REQUIRED_REAL_DATASETS:
        _require(
            ds in rates,
            f"architectural_observation.fallback_rate_by_dataset "
            f"missing dataset: {ds}",
        )
        value = float(rates[ds])
        _require(
            0.0 <= value <= 1.0,
            f"fallback rate for {ds} must be in [0, 1]",
        )
    _require(
        obs["post_hoc_tuning_applied"] is False,
        "architectural_observation.post_hoc_tuning_applied "
        "must be false",
    )


def _check_claims(data: dict[str, Any]) -> None:
    claims = data["claims"]
    for key in CLAIMS_TRUE:
        _require(
            claims[key] is True, f"claims.{key} must be true"
        )
    for key in CLAIMS_FALSE:
        _require(
            claims[key] is False, f"claims.{key} must be false"
        )


def _check_leakage(data: dict[str, Any]) -> None:
    lp = data["leakage_policy"]
    for key in LEAKAGE_KEYS:
        _require(
            lp[key] == "prohibited",
            f"leakage_policy.{key} must be prohibited",
        )


def _check_next_action(data: dict[str, Any]) -> None:
    nxt = data["next_action"]
    _require(
        nxt["class"] == "PAPER_REPORT_WRITING",
        "next_action.class mismatch",
    )
    _require(
        bool(nxt.get("value")),
        "next_action.value must be non-empty",
    )


# -- Integrity check -------------------------------------------------------


def _check_registry_hashes(data: dict[str, Any]) -> int:
    registry = data["artifact_registry"]
    checked = 0
    for entry in registry:
        rel = entry["path"]
        declared = entry["sha256"]
        _require(
            isinstance(rel, str) and len(rel) > 0,
            "artifact_registry.path must be a non-empty string",
        )
        _require(
            isinstance(declared, str) and len(declared) == 64,
            f"artifact_registry.sha256 for {rel} must be a 64-char hex",
        )
        target = REPO_ROOT / rel
        if not target.is_file():
            raise SystemExit(
                f"artifact_registry references a missing file: {rel}"
            )
        observed = _sha256_file(target)
        if observed != declared.upper():
            raise SystemExit(
                f"artifact hash mismatch for {rel}: "
                f"declared {declared}, observed {observed}"
            )
        checked += 1
    return checked


def main() -> int:
    data = _load()
    _check_top_level(data)
    _check_architecture(data)
    _check_tracks(data)
    _check_registry_paths(data)
    _check_gates(data)
    _check_real_dataset_status(data)
    _check_results(data)
    _check_architectural_observation(data)
    _check_claims(data)
    _check_leakage(data)
    _check_next_action(data)
    checked = _check_registry_hashes(data)

    print("[OK] Metadata-RAG readiness declaration")
    print(f"  declaration.status    = {data['declaration']['status']}")
    print(
        "  architecture_id       = "
        f"{data['architecture']['architecture_id']}"
    )
    print(
        "  fidelity_class        = "
        f"{data['architecture']['fidelity_class']}"
    )
    print(
        "  synthetic gates passed= "
        f"{sum(1 for g in REQUIRED_GATES if data['synthetic_acceptance']['gates'][g])}"
        f"/{len(REQUIRED_GATES)}"
    )
    print(
        "  real_dataset_status   = "
        f"{data['real_dataset_execution']['status']}"
    )
    print(f"  registry hashes check = {checked} files verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
