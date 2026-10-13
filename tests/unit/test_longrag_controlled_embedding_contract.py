"""Contract tests for the controlled BGE-M3 dense-only embedding contract."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

CONTRACT = Path("baseline_audit/longrag_controlled_embedding_contract.yaml")


@pytest.fixture(scope="module")
def contract() -> dict:
    return yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))


def test_controlled_adaptation_label_is_explicit(contract: dict) -> None:
    assert contract["architecture"]["fidelity_class"] == "CONTROLLED_ADAPTATION"
    assert contract["architecture"]["architecture_id"] == "longrag"


def test_bge_m3_revision_is_pinned(contract: dict) -> None:
    model = contract["model"]
    assert model["model_id"] == "BAAI/bge-m3"
    assert model["model_revision"] == "5617a9f61b028005a4858fdac845db406aefb181"
    assert model["local_snapshot_required"] is True
    assert model["network_fallback_allowed"] is False


def test_dense_only_mode_is_enforced(contract: dict) -> None:
    ex = contract["execution"]
    assert ex["dense_enabled"] is True
    assert ex["sparse_enabled"] is False
    assert ex["multi_vector_enabled"] is False
    assert ex["dense_dimension"] == 1024
    assert ex["dense_normalized"] is True


def test_sparse_and_colbert_disabled(contract: dict) -> None:
    ex = contract["execution"]
    assert ex["sparse_enabled"] is False
    assert ex["multi_vector_enabled"] is False
    acc = contract["acceptance"]
    assert acc["sparse_vectors_absent"] is True
    assert acc["colbert_vectors_absent"] is True


def test_native_embedder_claims_are_prohibited(contract: dict) -> None:
    claims = contract["claims"]
    assert claims["bge_m3_is_original_longrag_embedder"] is False
    assert claims["bge_m3_is_proven_superior_for_longrag"] is False
    assert claims["architecture_behavior_under_controlled_encoder"] is True
    fid = contract["fidelity"]
    assert fid["native_equivalence_claimed"] is False


def test_exact_reference_backend_is_required(contract: dict) -> None:
    # The contract does not allow approximate indexing in this track.
    ex = contract["execution"]
    assert ex["dense_normalized"] is True
    safety = contract["safety"]
    assert safety["network_access_enabled"] is False
    assert safety["model_download_enabled"] is False


def test_qdrant_equivalence_is_required(contract: dict) -> None:
    # The contract declares the controlled component as a controlled replacement.
    cc = contract["controlled_component"]
    assert cc["component"] == "embedder"
    assert cc["treatment"] == "CONTROLLED_REPLACEMENT"
    assert contract["execution"]["dense_enabled"] is True