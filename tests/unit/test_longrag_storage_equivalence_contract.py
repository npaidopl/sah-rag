"""Contract tests for LongRAG storage equivalence."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

CONTRACT = Path("baseline_audit/longrag_storage_equivalence_contract.yaml")


@pytest.fixture(scope="module")
def contract() -> dict:
    return yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))


def test_controlled_storage_is_dense_only_qdrant(contract: dict) -> None:
    cs = contract["controlled_storage"]
    assert cs["backend"] == "Qdrant"
    assert cs["vector_name"] == "dense"
    assert cs["dimension"] == 1024
    assert cs["distance"] == "COSINE"


def test_approximate_search_is_prohibited(contract: dict) -> None:
    cs = contract["controlled_storage"]
    assert cs["approximate_search_allowed"] is False
    assert cs["quantization_allowed"] is False
    assert cs["reranking_allowed"] is False
    rs = contract["reference_storage"]
    assert rs["approximate_search"] is False
    assert rs["candidate_pruning"] is False


def test_point_id_source_is_uuid5(contract: dict) -> None:
    ident = contract["identity"]
    assert "uuid5" in ident["point_id_source"]
    assert ident["point_id_prefix"] == "rag-longunit"
    assert ident["long_unit_id_preserved"] is True


def test_top_1_and_top_5_require_perfect_overlap(contract: dict) -> None:
    acc = contract["acceptance"]
    assert float(acc["top_1_overlap"]) == 1.0
    assert float(acc["top_5_overlap"]) == 1.0


def test_tie_policy_is_score_desc_then_id_asc(contract: dict) -> None:
    cmp_ = contract["comparison"]
    ref_tie = cmp_["tie_policy"]["reference"]
    qdr_tie = cmp_["tie_policy"]["qdrant_equal_score_normalization"]
    assert ref_tie == ["score_desc", "long_unit_id_asc"]
    assert qdr_tie == ["score_desc", "long_unit_id_asc"]


def test_pass_classification_is_neutral_for_controlled_track(contract: dict) -> None:
    cls = contract["classification"]
    assert cls["if_pass"] == "SEMANTICALLY_NEUTRAL_FOR_CONTROLLED_TRACK"
    assert cls["if_fail"] == "STORAGE_SUBSTITUTION_BLOCKED"


def test_fixture_declares_expected_dimensions(contract: dict) -> None:
    fx = contract["fixture"]
    assert fx["dataset_id"] == "infrastructure-pilot-synthetic-v1"
    assert fx["expected_documents"] == 12
    assert fx["expected_queries"] == 6
    assert fx["expected_relevant_documents_per_query"] == 2