"""Contract tests for the LongRAG synthetic acceptance contract."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

CONTRACT = Path("baseline_audit/longrag_synthetic_acceptance_contract.yaml")


@pytest.fixture(scope="module")
def contract() -> dict:
    return yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))


def test_no_actual_placeholders_remain() -> None:
    raw = CONTRACT.read_text(encoding="utf-8")
    assert "<actual>" not in raw


def test_long_unit_builder_is_real_function(contract: dict) -> None:
    lr = contract["longrag"]
    assert lr["long_unit_builder_function"] == (
        "baselines.longrag.construct_long_units"
    )
    assert lr["long_unit_identity_function"] == (
        "baselines.longrag.build_long_unit_id"
    )
    assert lr["canonical_expansion_function"] == (
        "baselines.longrag.expand_ranked_long_units_to_documents"
    )


def test_grouping_policy_and_target_size(contract: dict) -> None:
    lr = contract["longrag"]
    assert lr["grouping_policy"] == "adjacent_segments"
    assert lr["target_token_size"] == 256
    assert lr["token_counter"] == "whitespace_token_count"


def test_query_projection_excludes_ground_truth(contract: dict) -> None:
    qp = contract["longrag"]["query_projection"]
    assert "query_id" in qp["allowed_fields"]
    assert "query_text" in qp["allowed_fields"]
    for forbidden in ("doc_ids", "relevant_doc_ids", "answers", "ground_truth"):
        assert forbidden in qp["prohibited"]


def test_acceptance_gates_are_frozen(contract: dict) -> None:
    g = contract["acceptance_gates"]
    assert g["documents_loaded"] == 12
    assert g["queries_loaded"] == 6
    assert g["storage_equivalence_passed"] is True
    assert g["ground_truth_loaded_during_retrieval"] is False
    assert g["common_metrics_computed"] is False


def test_publishability_is_blocked(contract: dict) -> None:
    p = contract["publishability"]
    assert p["publishable_result"] is False
    assert p["native_faithful_reproduction"] is False
    assert p["controlled_adaptation"] is True


def test_native_track_is_not_eligible(contract: dict) -> None:
    a = contract["architecture"]
    assert a["controlled_track_eligible"] is True
    assert a["native_track_eligible"] is False
    assert a["native_track_potential_candidate"] is True