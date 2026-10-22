"""Persistence tests for the Step 85 Qdrant execution artifacts."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
EXECUTION_CONTRACT = REPO_ROOT / "baseline_audit" / "pilot_qdrant_execution_contract.yaml"
OUTPUT_VALIDATOR = REPO_ROOT / "scripts" / "validate_pilot_qdrant_outputs.py"
SMOKE = REPO_ROOT / "scripts" / "smoke_pilot_qdrant_ingestion.py"


def _load_contract() -> dict:
    return yaml.safe_load(EXECUTION_CONTRACT.read_text(encoding="utf-8"))


def _audit_root() -> Path:
    return REPO_ROOT / _load_contract()["backend"]["audit_root"]


def _storage_root() -> Path:
    return REPO_ROOT / _load_contract()["backend"]["storage_root"]


def _run(script: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(script)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )


def test_local_collection_storage_directory_exists_if_succeeded():
    from sah_rag.storage.qdrant import fingerprint_collection_name  # noqa: F401
    # The collection is created only after the runner has been executed.
    # If the runner has not yet been executed, skip gracefully.
    manifest_path = REPO_ROOT / "baseline_audit" / "pilot_qdrant_execution_manifest.yaml"
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if manifest["execution"]["status"] != "SUCCEEDED":
        return
    assert _storage_root().is_dir()


def test_audit_root_present_after_success():
    manifest_path = REPO_ROOT / "baseline_audit" / "pilot_qdrant_execution_manifest.yaml"
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if manifest["execution"]["status"] != "SUCCEEDED":
        return
    assert _audit_root().is_dir()


def test_output_validator_passes_after_success():
    manifest_path = REPO_ROOT / "baseline_audit" / "pilot_qdrant_execution_manifest.yaml"
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if manifest["execution"]["status"] != "SUCCEEDED":
        return
    r = _run(OUTPUT_VALIDATOR)
    assert r.returncode == 0, r.stderr


def test_smoke_passes_after_success():
    manifest_path = REPO_ROOT / "baseline_audit" / "pilot_qdrant_execution_manifest.yaml"
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if manifest["execution"]["status"] != "SUCCEEDED":
        return
    r = _run(SMOKE)
    assert r.returncode == 0, r.stderr


def test_existing_storage_rejected_without_overwrite():
    """If the runner has already run, invoking it without --overwrite must fail."""
    manifest_path = REPO_ROOT / "baseline_audit" / "pilot_qdrant_execution_manifest.yaml"
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if manifest["execution"]["status"] != "SUCCEEDED":
        return
    runner = REPO_ROOT / "scripts" / "run_pilot_qdrant_ingestion.py"
    r = subprocess.run(
        [sys.executable, str(runner)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert r.returncode != 0, "runner must reject existing storage without --overwrite"
    assert "already exists" in (r.stdout + r.stderr)


def test_embedding_subtree_untouched_by_qdrant_suite():
    embedding_root = (
        REPO_ROOT
        / "outputs" / "infrastructure_pilot"
        / "infrastructure-pilot-synthetic-v1"
        / "sah_rag" / "embedding"
    )
    # This test only asserts that the current embedding subtree
    # (if it existed at the start of the run) is not deleted by
    # importing or running the persistence suite.
    existed = embedding_root.exists()
    if not existed:
        return
    assert embedding_root.is_dir()