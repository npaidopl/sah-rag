"""Tests for the Metadata-RAG readiness declaration."""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DECLARATION = (
    REPO_ROOT
    / "baseline_audit"
    / "metadata_rag_readiness_declaration.yaml"
)
VALIDATOR = (
    REPO_ROOT
    / "scripts"
    / "validate_metadata_rag_readiness_declaration.py"
)


def _load() -> dict[str, Any]:
    raw = yaml.safe_load(DECLARATION.read_text(encoding="utf-8"))
    assert isinstance(raw, dict)
    return cast("dict[str, Any]", raw)


def _run_validator() -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(VALIDATOR)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


# --- Existence and validator ----------------------------------------------


def test_declaration_exists() -> None:
    assert DECLARATION.is_file()


def test_validator_accepts_declaration() -> None:
    result = _run_validator()
    assert result.returncode == 0, result.stderr
    assert (
        "[OK] Metadata-RAG readiness declaration" in result.stdout
    )
    assert "synthetic gates passed= 13/13" in result.stdout


# --- Structural -----------------------------------------------------------


def test_declaration_status_is_real_dataset_executed() -> None:
    data = _load()
    assert data["declaration"]["status"] == "REAL_DATASET_EXECUTED"


def test_architecture_identity_is_paper_derived_comparator() -> None:
    data = _load()
    arch = data["architecture"]
    assert arch["architecture_id"] == "metadata_rag"
    assert arch["fidelity_class"] == "PAPER_DERIVED_COMPARATOR"
    assert (
        arch["implementation_class"] == "PAPER_DERIVED_REIMPLEMENTATION"
    )


def test_native_track_is_deferred_with_reason() -> None:
    data = _load()
    native = data["tracks"]["native"]
    assert native["eligible"] is False
    assert native["reason"] == "DEFERRED_WITH_REASON"


def test_all_synthetic_gates_declared_true() -> None:
    data = _load()
    gates = data["synthetic_acceptance"]["gates"]
    for name, ok in gates.items():
        assert ok is True, name
    assert len(gates) == 13


def test_real_dataset_execution_is_executed() -> None:
    data = _load()
    rde = data["real_dataset_execution"]
    assert rde["status"] == "EXECUTED"
    assert set(rde["executed_scope"]) == {
        "clerk",
        "cuad",
        "greek_bar_bench",
        "lara",
    }


def test_results_block_present() -> None:
    data = _load()
    results = data["results"]
    assert results["primary_metric"] == "macro_document_recall_at_10"
    assert set(results["per_dataset"]) == {
        "clerk",
        "cuad",
        "greek_bar_bench",
        "lara",
    }
    assert set(results["three_track_comparison"]) == {
        "longrag",
        "metadata_rag",
        "sah_rag",
    }


def test_architectural_observation_present() -> None:
    data = _load()
    obs = data["architectural_observation"]
    assert set(obs["fallback_rate_by_dataset"]) == {
        "clerk",
        "cuad",
        "greek_bar_bench",
        "lara",
    }
    assert obs["post_hoc_tuning_applied"] is False


def test_embedder_is_pinned_bge_m3_dense_only() -> None:
    data = _load()
    emb = data["real_dataset_execution"]["embedder"]
    assert emb["model_id"] == "BAAI/bge-m3"
    assert (
        emb["model_revision"]
        == "5617a9f61b028005a4858fdac845db406aefb181"
    )
    assert emb["mode"] == "dense_only"


def test_protocol_cutoffs_and_primary_metric() -> None:
    data = _load()
    protocol = data["real_dataset_execution"]["protocol"]
    assert tuple(protocol["cutoffs"]) == (1, 5, 10, 20, 50, 100)
    assert protocol["primary_metric"] == "macro_document_recall_at_10"
    assert protocol["generation_enabled"] is False
    assert protocol["llm_judge_enabled"] is False


def test_claims_are_conservative() -> None:
    data = _load()
    claims = data["claims"]
    assert claims["synthetic_validated"] is True
    assert claims["paper_derived_comparator"] is True
    assert claims["controlled_adaptation"] is True
    assert claims["faithful_reproduction"] is False
    assert claims["paper_effectiveness_reproduced"] is False
    assert claims["native_equivalence"] is False
    assert claims["real_dataset_validated"] is True
    assert claims["publishable_result"] is False
    assert (
        claims["synthetic_validation_is_paper_level_evidence"] is False
    )


def test_leakage_policy_prohibits_all_five() -> None:
    data = _load()
    lp = data["leakage_policy"]
    for key in (
        "judgments_as_metadata",
        "reference_answers_as_metadata",
        "evaluation_labels_as_metadata",
        "generated_answers_as_metadata",
        "post_retrieval_annotations_as_metadata",
    ):
        assert lp[key] == "prohibited", key


# --- Registry integrity ----------------------------------------------------


def test_registry_covers_all_production_modules() -> None:
    data = _load()
    registry = data["artifact_registry"]
    paths = {entry["path"] for entry in registry}
    required = {
        "src/rag_system/baselines/metadata_rag_metadata.py",
        "src/rag_system/baselines/metadata_rag_adapter.py",
        "src/rag_system/baselines/metadata_rag_state.py",
        "src/rag_system/baselines/metadata_rag_persistence.py",
        "src/rag_system/baselines/metadata_rag_embedder.py",
        "src/rag_system/baselines/metadata_rag_reference_index.py",
        "src/rag_system/baselines/metadata_rag_retrieval.py",
        "src/rag_system/baselines/metadata_rag_expansion.py",
        "src/rag_system/baselines/metadata_rag_evaluation_case.py",
    }
    assert required.issubset(paths)


def test_every_registry_entry_hash_matches_ondisk() -> None:
    data = _load()
    for entry in data["artifact_registry"]:
        rel = entry["path"]
        target = REPO_ROOT / rel
        assert target.is_file(), rel
        assert _sha256_file(target) == entry["sha256"].upper(), rel
