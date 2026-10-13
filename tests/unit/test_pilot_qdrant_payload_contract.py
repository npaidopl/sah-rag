"""Tests for the pilot Qdrant payload contract."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
PATH = REPO_ROOT / "baseline_audit" / "pilot_qdrant_payload_contract.yaml"
VALIDATOR = REPO_ROOT / "scripts" / "validate_pilot_qdrant_payload_contract.py"

REQUIRED = (
    "chunk_id",
    "parent_id",
    "doc_id",
    "dataset",
    "chunk_index",
    "parent_chunk_index",
    "text_normalized",
    "normalized_char_start",
    "normalized_char_end",
    "token_count",
    "model_id",
    "model_revision",
    "embedding_config_hash",
    "embedding_shard_id",
)


def _load() -> dict:
    return yaml.safe_load(PATH.read_text(encoding="utf-8"))


def _run() -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(VALIDATOR)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )


def test_validator_accepts_payload_contract():
    r = _run()
    assert r.returncode == 0, r.stderr


def test_required_fields_exact():
    required = _load()["payload"]["required_fields"]
    for name in REQUIRED:
        assert name in required, name


def test_dataset_identity_exact():
    required = _load()["payload"]["required_fields"]
    assert required["dataset"]["exact_value"] == "infrastructure-pilot-synthetic-v1"


def test_model_identity_and_revision_exact():
    required = _load()["payload"]["required_fields"]
    assert required["model_id"]["exact_value"] == "BAAI/bge-m3"
    assert (
        required["model_revision"]["exact_value"]
        == "5617a9f61b028005a4858fdac845db406aefb181"
    )


def test_checksum_format_enforced():
    required = _load()["payload"]["required_fields"]
    assert required["embedding_config_hash"]["format"] == "lowercase_sha256"


def test_evaluation_answer_and_secret_fields_prohibited():
    prohibited = set(_load()["payload"]["prohibited_fields"])
    for name in (
        "generated_answer",
        "reference_answer",
        "relevance",
        "judgment",
        "API_key",
        "access_token",
    ):
        assert name in prohibited, name


def test_spec_divergences_documented():
    notes = _load()["payload"]["spec_requested_but_not_produced"]
    assert "dataset_id" in notes
    assert "source_order" in notes
    assert "content_sha256" in notes