"""Tests for the pilot BGE-M3 execution contract."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
PATH = REPO_ROOT / "baseline_audit" / "pilot_embedding_execution_contract.yaml"
VALIDATOR = REPO_ROOT / "scripts" / "validate_pilot_embedding_execution_contract.py"

FROZEN_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"


def _load() -> dict:
    return yaml.safe_load(PATH.read_text(encoding="utf-8"))


def _run() -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(VALIDATOR)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )


# --- contract and approvals (5) ---------------------------------


def test_validator_accepts_execution_contract():
    r = _run()
    assert r.returncode == 0, r.stderr


def test_contract_status_is_approved_for_bounded_execution():
    assert _load()["contract"]["status"] == "APPROVED_FOR_BOUNDED_EXECUTION"


def test_only_model_load_and_embedding_enabled():
    execution = _load()["execution"]
    assert execution["model_load_enabled"] is True
    assert execution["embedding_enabled"] is True
    for flag in (
        "model_download_enabled",
        "network_access_enabled",
        "hosted_requests_enabled",
        "qdrant_collection_creation_enabled",
        "vector_ingestion_enabled",
        "retrieval_enabled",
        "evaluation_enabled",
        "generation_enabled",
    ):
        assert execution[flag] is False, flag


def test_exact_local_snapshot_path_pinned():
    model = _load()["model"]
    assert model["model_revision"] == FROZEN_REVISION
    assert model["snapshot_path"].endswith(FROZEN_REVISION)
    assert model["local_files_only"] is True
    assert model["network_fallback_allowed"] is False


def test_input_count_is_twelve():
    assert _load()["source"]["expected_child_chunk_count"] == 12


# --- benchmark (5) ----------------------------------------------


def test_batch_sizes_are_one_two_four():
    assert _load()["benchmark"]["batch_sizes"] == [1, 2, 4]


def test_warmup_count_is_one():
    assert _load()["benchmark"]["warmup_run_count"] == 1


def test_three_measured_runs_per_batch_size():
    assert _load()["benchmark"]["measured_run_count_per_batch_size"] == 3


def test_raw_runs_retained():
    assert _load()["benchmark"]["raw_runs_retained"] is True


def test_linear_extrapolation_prohibited():
    assert _load()["benchmark"]["linear_extrapolation_allowed"] is False


# --- embedding validation (6) -----------------------------------


def test_dense_dimension_is_1024():
    assert _load()["model"]["dense_dimension"] == 1024


def test_dense_vector_count_is_12():
    assert _load()["acceptance_gates"]["dense_vector_count"] == 12


def test_sparse_vector_count_is_12():
    assert _load()["acceptance_gates"]["sparse_vector_count"] == 12


def test_multi_vector_disabled():
    assert _load()["model"]["multi_vector_enabled"] is False
    assert _load()["model"]["init_params"]["return_colbert_vecs"] is False


def test_offline_flags_set():
    offline = _load()["offline_enforcement"]
    assert offline["HF_HUB_OFFLINE"] == "1"
    assert offline["TRANSFORMERS_OFFLINE"] == "1"
    assert offline["HF_DATASETS_OFFLINE"] == "1"
    assert offline["OMP_NUM_THREADS"] == "1"
    assert offline["MKL_NUM_THREADS"] == "1"


def test_ram_guard_requires_at_least_two_gib():
    assert _load()["resource_limits"]["minimum_available_ram_bytes"] >= 2147483648


# --- persistence / lineage (4) ----------------------------------


def test_required_materialization_files_present():
    materialization = _load()["materialization"]
    required = materialization["required_files"]
    for name in (
        "shards/shard-00000/header.json",
        "shards/shard-00000/records.jsonl",
        "shards/shard-00000/shard.manifest.json",
        "embedding.catalog.json",
        "embedding.manifest.json",
        "checksums.json",
    ):
        assert name in required
    assert materialization["shards_subdir"] == "shards"


def test_shard_count_is_one():
    assert _load()["materialization"]["expected_shard_count"] == 1


def test_lineage_gates_enabled():
    gates = _load()["acceptance_gates"]
    assert gates["parent_lineage_preserved"] is True
    assert gates["document_lineage_preserved"] is True
    assert gates["chunk_ids_unique"] is True


def test_publishability_flags_disabled():
    pub = _load()["publishability"]
    assert pub["publishable_result"] is False
    assert pub["included_in_final_comparison"] is False
    assert pub["retrieval_effectiveness_claimed"] is False