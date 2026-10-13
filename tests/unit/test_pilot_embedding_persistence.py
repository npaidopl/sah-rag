"""Unit tests for the pilot embedding persistence contract."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONTRACT = REPO_ROOT / "baseline_audit" / "pilot_embedding_execution_contract.yaml"
MATERIALIZATION_RUNNER = REPO_ROOT / "scripts" / "run_pilot_embedding_materialization.py"
OUTPUT_VALIDATOR = REPO_ROOT / "scripts" / "validate_pilot_embedding_outputs.py"


def _contract() -> dict:
    return yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))


def _load_runner():
    spec = importlib.util.spec_from_file_location("_pilot_mat2", MATERIALIZATION_RUNNER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_required_files_include_shard_and_catalog_and_checksums():
    required = _contract()["materialization"]["required_files"]
    assert "shards/shard-00000/header.json" in required
    assert "shards/shard-00000/records.jsonl" in required
    assert "shards/shard-00000/shard.manifest.json" in required
    assert "embedding.catalog.json" in required
    assert "embedding.manifest.json" in required
    assert "checksums.json" in required


def test_no_overwrite_of_existing_outputs():
    runner = _load_runner()
    import inspect

    sig = inspect.signature(runner.main)
    assert sig is not None
    source = inspect.getsource(runner.main)
    assert "shard directory already exists" in source


def test_output_validator_exposes_main_entrypoint():
    spec = importlib.util.spec_from_file_location("_pilot_out_val", OUTPUT_VALIDATOR)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    assert callable(module.main)


def test_writer_files_recorded_in_contract():
    writer = _contract()["materialization"]["writer"]
    assert writer["function"].endswith("write_embedding_shard")
    assert "records.jsonl" in writer["files_per_shard"]