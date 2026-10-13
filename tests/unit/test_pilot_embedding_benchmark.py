"""Unit tests for the pilot BGE-M3 CPU benchmark logic."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER = REPO_ROOT / "scripts" / "run_pilot_embedding_benchmark.py"
VALIDATOR = REPO_ROOT / "scripts" / "validate_pilot_embedding_benchmark_report.py"
REPORT = REPO_ROOT / "baseline_audit" / "pilot_embedding_benchmark_report.yaml"


def _load_runner():
    spec = importlib.util.spec_from_file_location("_pilot_benchmark", RUNNER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_report() -> dict:
    return yaml.safe_load(REPORT.read_text(encoding="utf-8"))


def _run_validator() -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(VALIDATOR)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )


def test_validator_accepts_planned_report():
    r = _run_validator()
    assert r.returncode == 0, r.stderr


def test_report_state_matches_status():
    """The report must have a PLANNED or SUCCEEDED shape consistent with its status."""
    report = _load_report()
    status = report["benchmark"]["status"]
    if status == "PLANNED":
        assert report["results"] == {}
        assert report["selection"]["selected_batch_size"] is None
        assert report["execution_state"]["model_loaded"] is False
    elif status == "SUCCEEDED":
        assert set(report["results"].keys()) == {
            "batch_size_1", "batch_size_2", "batch_size_4",
        }
        assert report["selection"]["selected_batch_size"] in (1, 2, 4)
        assert report["execution_state"]["model_loaded"] is True
    else:
        raise AssertionError("unexpected benchmark status: " + str(status))


def test_median_function_matches_statistics_median():
    runner = _load_runner()
    assert runner.median([1.0, 2.0, 3.0]) == 2.0
    assert runner.median([1.0, 3.0]) == 2.0
    assert runner.median([5.0]) == 5.0


def test_sparse_normalization_sorts_ascending_and_drops_nonpositive():
    runner = _load_runner()
    indices, values = runner.normalize_sparse({"3": 0.5, "1": 0.2, "2": 0.0, "5": 0.9})
    assert indices == [1, 3, 5]
    assert values == [0.2, 0.5, 0.9]


def test_select_batch_size_prefers_highest_with_no_worse_throughput():
    runner = _load_runner()
    results = {
        1: {"status": "OK", "median_records_per_second": 5.0},
        2: {"status": "OK", "median_records_per_second": 6.0},
        4: {"status": "OK", "median_records_per_second": 6.5},
    }
    selected, _reason = runner.select_batch_size(results)
    assert selected == 4


def test_select_batch_size_avoids_slower_larger_batch():
    runner = _load_runner()
    results = {
        1: {"status": "OK", "median_records_per_second": 6.0},
        2: {"status": "OK", "median_records_per_second": 5.0},
        4: {"status": "OK", "median_records_per_second": 4.0},
    }
    selected, _reason = runner.select_batch_size(results)
    assert selected == 1


def test_select_batch_size_returns_none_when_all_failed():
    runner = _load_runner()
    results = {
        1: {"status": "FAILED"},
        2: {"status": "FAILED"},
        4: {"status": "FAILED"},
    }
    selected, _reason = runner.select_batch_size(results)
    assert selected is None


def test_linear_extrapolation_prohibited_in_report_template():
    interpretation = _load_report()["interpretation"]
    assert interpretation["linear_extrapolation_allowed"] is False