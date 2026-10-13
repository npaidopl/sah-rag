"""Unit tests for scripts/run_longrag_architecture_freeze_batch.py."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = (
    REPO_ROOT / "scripts" / "run_longrag_architecture_freeze_batch.py"
)


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "longrag_full_track_freeze", SCRIPT_PATH
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["longrag_full_track_freeze"] = module
    spec.loader.exec_module(module)
    return module


def test_expected_datasets_exact() -> None:
    m = _load_module()
    assert tuple(m.EXPECTED_DATASETS) == (
        "ruler",
        "loong",
        "long_bench_v2",
        "lara",
        "greek_bar_bench",
        "cuad",
        "clerk",
    )


def test_embedding_model_and_revision_pinned() -> None:
    m = _load_module()
    assert m.EMBEDDING_MODEL_ID == "BAAI/bge-m3"
    assert (
        m.EMBEDDING_MODEL_REVISION
        == "5617a9f61b028005a4858fdac845db406aefb181"
    )


def test_component_disposition_has_defining_components() -> None:
    m = _load_module()
    disposition = m.build_component_disposition()
    components = disposition["components"]
    roles = {name: spec["role"] for name, spec in components.items()}
    assert roles["long_document_grouping"] == "ARCHITECTURE_DEFINING"
    assert roles["dense_only_retrieval"] == "ARCHITECTURE_DEFINING"


def test_allowed_roles_only() -> None:
    m = _load_module()
    disposition = m.build_component_disposition()
    allowed = {
        "ARCHITECTURE_DEFINING",
        "ARCHITECTURE_SUPPORTING",
        "OPTIONAL_DOWNSTREAM",
        "NOT_PRESENT",
    }
    for name, spec in disposition["components"].items():
        assert spec["role"] in allowed, (name, spec["role"])


def test_architecture_freeze_defining_set() -> None:
    m = _load_module()
    freeze = m.build_architecture_freeze()
    assert set(freeze["architecture_defining_components"]) == {
        "long_document_grouping",
        "dense_only_retrieval",
    }
    assert freeze["retrieval_semantics"]["sparse_enabled"] is False
    assert freeze["retrieval_semantics"]["hybrid_enabled"] is False
    assert freeze["retrieval_semantics"]["reranking_enabled"] is False
    assert freeze["storage_semantics"]["dimension"] == 1024
    assert freeze["storage_semantics"]["distance"] == "COSINE"


def test_embedding_profile_primary_and_reference() -> None:
    m = _load_module()
    profile = m.build_embedding_profile()
    assert profile["execution"]["environment"] == "Google Colab GPU"
    assert profile["execution"]["dtype_primary"] == "float16"
    ref = profile["bounded_reference_profile"]
    assert ref["enabled"] is True
    assert ref["dtype"] == "float32"
    assert ref["subset_size_per_dataset"] == 100
    assert (
        profile["contract"]["synthetic_acceptance_contract_untouched"]
        is True
    )


def test_plan_phase_order() -> None:
    m = _load_module()
    plan = m.build_plan()
    assert plan["order"] == ["L1", "L2", "L3", "L4", "L5", "L6", "L7"]
    for phase in plan["order"]:
        assert phase in plan["phases"], phase