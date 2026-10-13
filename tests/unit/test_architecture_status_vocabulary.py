"""Tests for the architecture status vocabulary."""

from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
VOCAB_PATH = (
    REPO_ROOT
    / "baseline_audit"
    / "architecture_status_vocabulary.yaml"
)


def _load() -> dict:
    with VOCAB_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_artifact_type() -> None:
    d = _load()
    assert d["artifact_type"] == "architecture_status_vocabulary"


def test_ready_for_data_is_not_dataset_ready() -> None:
    d = _load()
    rfd = d["status_classes"]["READY_FOR_DATA"]
    # READY_FOR_DATA must not be counted in dataset_ready_count.
    assert "dataset_ready_count" not in (rfd.get("counted_in") or [])


def test_synthetic_validated_not_counted_as_dataset_ready() -> None:
    d = _load()
    sv = d["status_classes"]["SYNTHETIC_VALIDATED"]
    assert "dataset_ready_count" not in (sv.get("counted_in") or [])


def test_deferred_counted_in_deferred_count() -> None:
    d = _load()
    df = d["status_classes"]["DEFERRED"]
    assert "deferred_count" in df["counted_in"]


def test_dataset_ready_counted_in_dataset_ready_count() -> None:
    d = _load()
    dr = d["status_classes"]["DATASET_READY"]
    assert "dataset_ready_count" in dr["counted_in"]


def test_counter_source_is_records() -> None:
    d = _load()
    for name, spec in d["counter_definitions"].items():
        assert spec["source"] == "ARCHITECTURE_READINESS_RECORDS", (
            f"counter {name} must be sourced from records"
        )
        assert spec["manual_counter_edits_allowed"] is False, (
            f"counter {name} must forbid manual edits"
        )