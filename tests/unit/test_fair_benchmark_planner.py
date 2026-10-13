"""Tests for the Step 93 fair benchmark planner."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
PLANNER = REPO_ROOT / "scripts" / "plan_fair_benchmark.py"
MANIFEST = REPO_ROOT / "baseline_audit" / "fair_benchmark_manifest.yaml"


def _load_planner():
    spec = importlib.util.spec_from_file_location("_fair_planner", PLANNER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_planner_exists():
    assert PLANNER.exists()


def test_manifest_state_after_run():
    m = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    assert m["execution"]["status"] in ("PLANNED","SUCCEEDED","INTERRUPTED")


def test_manifest_execution_flags_disabled():
    m = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    a = m["approval"]
    for flag in ("architecture_implementation_approved","tokenizer_execution_approved",
                 "model_execution_approved","embedding_execution_approved",
                 "retrieval_execution_approved","evaluation_execution_approved"):
        assert a[flag] is False, flag


def test_next_gate_correct():
    ng = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))["next_gate"]
    assert ng["required_step"] == "per_architecture_readiness_gap_registry"
    assert ng["approved"] is False


def test_planner_does_not_load_model_or_tokenizer():
    src = PLANNER.read_text(encoding="utf-8")
    for token in (
        "BGEM3FlagModel",
        "FlagEmbedding",
        "AutoTokenizer",
        "from_pretrained",
        "QdrantClient",
    ):
        assert token not in src, token


def test_planner_does_not_execute_pipeline():
    src = PLANNER.read_text(encoding="utf-8")
    for token in ("run_pilot_parser_chunker(","run_pilot_embedding_materialization(",
                  "run_pilot_qdrant_ingestion(","run_pilot_hybrid_retrieval(",
                  "run_pilot_retrieval_evaluation("):
        assert token not in src, token