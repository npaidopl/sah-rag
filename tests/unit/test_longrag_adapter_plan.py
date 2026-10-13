"""Tests for the LongRAG adapter planning contract."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

SCRIPT_PATH = (
    Path(__file__).parents[2]
    / "scripts"
    / "validate_longrag_adapter_plan.py"
)


def load_script() -> ModuleType:
    """Load the plan validator."""
    specification = (
        importlib.util.spec_from_file_location(
            "longrag_adapter_plan",
            SCRIPT_PATH,
        )
    )
    if (
        specification is None
        or specification.loader is None
    ):
        raise RuntimeError(
            "Could not load LongRAG plan validator."
        )
    module = importlib.util.module_from_spec(
        specification
    )
    specification.loader.exec_module(
        module
    )
    return module


def test_plan_validator_accepts_current_contract() -> None:
    """The current planning contract must validate."""
    module = load_script()
    module.main()


def test_expected_commit_is_pinned() -> None:
    """The audited LongRAG source revision must be pinned."""
    module = load_script()
    assert module.EXPECTED_COMMIT == (
        "ba92f95"
    )


def test_metric_contract_is_retrieval_only() -> None:
    """The adapter metrics must contain retrieval metrics only."""
    module = load_script()
    assert {
        "recall_at_k",
        "precision_at_k",
        "reciprocal_rank",
        "ndcg_at_k",
        "hit_at_k",
    } == module.REQUIRED_METRICS
