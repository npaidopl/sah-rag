"""Tests for the LongRAG readiness declaration."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

DECL = Path("baseline_audit/longrag_readiness_declaration.yaml")

VALIDATED_FLAGS = (
    "canonical_input_validated",
    "long_unit_construction_validated",
    "controlled_embedding_validated",
    "persistent_state_validated",
    "state_reopen_validated",
    "retrieval_validated",
    "storage_equivalence_validated",
    "canonical_identity_expansion_validated",
    "evaluation_case_mapping_validated",
)


@pytest.fixture(scope="module")
def decl() -> dict:
    return yaml.safe_load(DECL.read_text(encoding="utf-8"))


def test_passed_state_is_consistent(decl: dict) -> None:
    rd = decl["readiness"]
    assert rd["status"] == "SYNTHETIC_ACCEPTANCE_PASSED"
    for k in VALIDATED_FLAGS:
        assert rd[k] is True, f"{k} must be true after acceptance passed"
    assert rd["dataset_ready"] is True
    assert rd["generation_disabled"] is True


def test_native_fidelity_is_not_claimed(decl: dict) -> None:
    claims = decl["claims"]
    assert claims["faithful_reproduction"] is False
    assert claims["native_embedder_equivalence"] is False
    assert claims["controlled_adaptation"] is True
    assert claims["publishable_result"] is False


def test_track_status_is_controlled_only(decl: dict) -> None:
    tr = decl["track"]
    assert tr["controlled_track_eligible"] is True
    assert tr["native_track_eligible"] is False
    assert tr["native_track_potential_candidate"] is True


def test_next_action_is_dataset_resource_planning(decl: dict) -> None:
    na = decl["next_action"]
    assert na["class"] == "DATASET_RESOURCE_PLANNING"
    assert "LongRAG" in na["value"]