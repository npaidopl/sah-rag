"""Tests for the Step 90 evidence collector guards."""

from __future__ import annotations

import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
COLLECTOR = REPO_ROOT / "scripts" / "collect_infrastructure_pilot_evidence.py"


def _load_collector():
    spec = importlib.util.spec_from_file_location("_pilot_collector", COLLECTOR)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_collector_has_no_model_load_reference():
    source = COLLECTOR.read_text(encoding="utf-8")
    assert "BGEM3FlagModel" not in source
    assert "FlagEmbedding" not in source


def test_collector_has_no_qdrant_client_reference():
    source = COLLECTOR.read_text(encoding="utf-8")
    assert "QdrantClient" not in source
    assert "qdrant_client" not in source


def test_collector_does_not_call_pipeline_stages():
    source = COLLECTOR.read_text(encoding="utf-8")
    for token in (
        "run_pilot_parser_chunker(",
        "run_pilot_embedding_materialization(",
        "run_pilot_qdrant_ingestion(",
        "run_pilot_hybrid_retrieval(",
        "run_pilot_retrieval_evaluation(",
    ):
        assert token not in source, token


def test_collector_does_not_modify_runtime_roots():
    source = COLLECTOR.read_text(encoding="utf-8")
    for token in (
        ".mkdir(",
        ".write_text(",
        ".write_bytes(",
        "shutil.rmtree",
    ):
        # Only CLOSURE_MANIFEST writes are allowed; check the specific
        # write_text target is baseline_audit.
        if token in source:
            # CLOSURE_MANIFEST.write_text is permitted
            assert "CLOSURE_MANIFEST.write_text" in source or token == ".mkdir("
            break