"""Tests for L1a amendment on the LongRAG full-track freeze."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = (
    REPO_ROOT / "scripts" / "run_longrag_architecture_freeze_batch.py"
)


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "longrag_full_track_freeze_l1a", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["longrag_full_track_freeze_l1a"] = module
    spec.loader.exec_module(module)
    return module


def test_l1a_constants() -> None:
    m = _load()
    assert m.LONG_UNIT_TARGET_TOKEN_SIZE == 4096
    assert m.LONG_UNIT_CONSTRUCTION_POLICY == (
        "adjacent_document_segments"
    )
    assert m.LONG_UNIT_TOKEN_COUNTER == "pinned_bge_m3_tokenizer"
    assert m.CONTRACT_VERSION == "0.3.0"


def test_architecture_freeze_uses_l1a_semantics() -> None:
    m = _load()
    freeze = m.build_architecture_freeze()
    semantics = freeze["long_unit_semantics"]
    assert semantics["target_token_size"] == 4096
    assert semantics["construction_policy"] == (
        "adjacent_document_segments"
    )
    assert semantics["oversize_handling"] == (
        "split_into_target_size_windows_via_pinned_bge_m3_tokenizer"
    )
    assert semantics["tokenizer_id"] == "BAAI/bge-m3"
    assert semantics["tokenizer_revision"] == (
        "5617a9f61b028005a4858fdac845db406aefb181"
    )


def test_disposition_has_oversize_component() -> None:
    m = _load()
    disposition = m.build_component_disposition()
    components = disposition["components"]
    assert components["oversize_handling"]["role"] == (
        "ARCHITECTURE_SUPPORTING"
    )
    assert components["tokenizer_bound_segment_tokens"]["role"] == (
        "ARCHITECTURE_SUPPORTING"
    )