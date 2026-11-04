"""Unit tests for orchestration.run_all (offline, mock-based)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from orchestration.run_all import (
    RunAllState,
    StageState,
    _build_scheduler_manifest,
    _check_env_var,
    _check_model_dir,
    get_output_root,
    load_master_manifest,
    load_or_create_state,
    save_state,
)


# =====================================================================
# State management
# =====================================================================


def test_state_load_create_fresh(tmp_path):
    state_path = tmp_path / "state.json"
    state = load_or_create_state(state_path)
    assert state.version == "1.0"
    assert state.started_at != ""
    assert state.finished_at is None
    assert state.stages == {}


def test_state_save_and_load_roundtrip(tmp_path):
    state_path = tmp_path / "state.json"

    state = RunAllState(started_at="2026-11-04T00:00:00")
    state.set_stage("0_preflight", StageState(
        done=True,
        started_at="2026-11-04T00:00:01",
        finished_at="2026-11-04T00:00:02",
        duration_s=1.0,
    ))
    save_state(state, state_path)

    loaded = load_or_create_state(state_path)
    assert loaded.version == "1.0"
    assert loaded.started_at == "2026-11-04T00:00:00"
    assert "0_preflight" in loaded.stages
    assert loaded.stages["0_preflight"]["done"] is True


def test_state_get_set_stage(tmp_path):
    state = RunAllState(started_at="2026-11-04T00:00:00")

    # Empty by default
    st = state.get_stage("1_optimal")
    assert st.done is False
    assert st.error is None

    # Set and retrieve
    state.set_stage("1_optimal", StageState(done=True, duration_s=10.5))
    st = state.get_stage("1_optimal")
    assert st.done is True
    assert st.duration_s == 10.5


def test_state_save_creates_parent_dirs(tmp_path):
    state_path = tmp_path / "deep" / "nested" / "state.json"
    state = RunAllState(started_at="2026-11-04T00:00:00")
    save_state(state, state_path)
    assert state_path.exists()


def test_state_save_is_atomic(tmp_path):
    """No .tmp file should remain after successful save."""
    state_path = tmp_path / "state.json"
    state = RunAllState(started_at="2026-11-04T00:00:00")
    save_state(state, state_path)
    tmp_files = list(tmp_path.glob("*.tmp"))
    assert len(tmp_files) == 0


# =====================================================================
# Master manifest
# =====================================================================


def test_load_master_manifest_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_master_manifest(tmp_path / "nonexistent.yaml")


def test_load_master_manifest_basic(tmp_path):
    manifest_path = tmp_path / "manifest.yaml"
    manifest_data = {
        "version": "1.0",
        "global": {
            "output_root": str(tmp_path / "out"),
            "stop_on_fatal": True,
        },
        "preflight": {"check_env": ["RAG_CANONICAL_ROOT"]},
        "controlled_adaptation": {"enabled": True, "baselines": ["a"], "datasets": ["b"]},
    }
    manifest_path.write_text(yaml.safe_dump(manifest_data), encoding="utf-8")

    loaded = load_master_manifest(manifest_path)
    assert loaded["version"] == "1.0"
    assert loaded["global"]["output_root"] == str(tmp_path / "out")
    assert loaded["controlled_adaptation"]["enabled"] is True


def test_get_output_root_missing(tmp_path):
    with pytest.raises(RuntimeError, match="output_root is required"):
        get_output_root({"global": {}})


def test_get_output_root_ok(tmp_path):
    path = get_output_root({"global": {"output_root": str(tmp_path)}})
    assert path == tmp_path


# =====================================================================
# Env var checking
# =====================================================================


def test_check_env_var_set(monkeypatch):
    monkeypatch.setenv("TEST_VAR_OK", "somevalue")
    ok, msg = _check_env_var("TEST_VAR_OK", required=True)
    assert ok is True
    assert "TEST_VAR_OK" in msg


def test_check_env_var_missing_required(monkeypatch):
    monkeypatch.delenv("TEST_VAR_MISSING", raising=False)
    ok, msg = _check_env_var("TEST_VAR_MISSING", required=True)
    assert ok is False
    assert "NOT SET" in msg
    assert "REQUIRED" in msg


def test_check_env_var_missing_optional(monkeypatch):
    monkeypatch.delenv("TEST_VAR_OPT", raising=False)
    ok, msg = _check_env_var("TEST_VAR_OPT", required=False)
    assert ok is True  # optional missing is OK
    assert "NOT SET" in msg
    assert "optional" in msg


# =====================================================================
# Model dir checking (with fallback)
# =====================================================================


def test_check_model_dir_env_var_unset_no_default(monkeypatch):
    monkeypatch.delenv("TEST_MODEL_ROOT", raising=False)
    ok, msg = _check_model_dir("TEST_MODEL_ROOT", 1.0, default_path=None)
    assert ok is False
    assert "NOT SET" in msg
    assert "required" in msg


def test_check_model_dir_env_var_unset_with_default_missing(monkeypatch, tmp_path):
    monkeypatch.delenv("TEST_MODEL_ROOT", raising=False)
    default = tmp_path / "nonexistent_model"
    ok, msg = _check_model_dir("TEST_MODEL_ROOT", 1.0, default_path=str(default))
    assert ok is False
    assert "does not exist" in msg


def test_check_model_dir_env_var_set_but_too_small(monkeypatch, tmp_path):
    """Set env var to a dir with 1 file too small for expected size."""
    small_dir = tmp_path / "small_model"
    small_dir.mkdir()
    (small_dir / "tiny.bin").write_bytes(b"x" * 1000)  # 1KB

    monkeypatch.setenv("TEST_MODEL_ROOT", str(small_dir))
    ok, msg = _check_model_dir("TEST_MODEL_ROOT", 1.0, default_path=None)
    assert ok is False
    assert "GB" in msg
    assert "expected" in msg


def test_check_model_dir_ok(monkeypatch, tmp_path):
    """Set env var to a dir with a file above threshold."""
    big_dir = tmp_path / "big_model"
    big_dir.mkdir()
    # Create ~1.05 GB (above 1.0 * 0.9 threshold)
    target_bytes = int(1.05 * 1024 ** 3)
    (big_dir / "weights.bin").write_bytes(b"x" * target_bytes)

    monkeypatch.setenv("TEST_MODEL_ROOT", str(big_dir))
    ok, msg = _check_model_dir("TEST_MODEL_ROOT", 1.0, default_path=None)
    assert ok is True
    assert "GB" in msg


def test_check_model_dir_fallback_to_default(monkeypatch, tmp_path):
    """Env var unset, default has enough bytes."""
    big_dir = tmp_path / "fallback_model"
    big_dir.mkdir()
    target_bytes = int(1.05 * 1024 ** 3)
    (big_dir / "weights.bin").write_bytes(b"x" * target_bytes)

    monkeypatch.delenv("TEST_MODEL_ROOT", raising=False)
    ok, msg = _check_model_dir("TEST_MODEL_ROOT", 1.0, default_path=str(big_dir))
    assert ok is True
    assert "fallback" in msg


# =====================================================================
# Scheduler manifest generation
# =====================================================================


def test_build_scheduler_manifest_creates_file(tmp_path):
    manifest_path = tmp_path / "generated.yaml"
    out_root = tmp_path / "out"

    _build_scheduler_manifest(
        modes=["controlled"],
        baselines=["bm25", "splade"],
        datasets=["clerk", "cuad"],
        top_k=10,
        output_root=out_root,
        manifest_path=manifest_path,
    )

    assert manifest_path.exists()
    data = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    assert data["modes"] == ["controlled"]
    assert data["baselines"] == ["bm25", "splade"]
    assert data["datasets"] == ["clerk", "cuad"]
    assert data["top_k"] == 10
    assert data["output_root"] == str(out_root)


def test_build_scheduler_manifest_creates_parent_dirs(tmp_path):
    manifest_path = tmp_path / "deep" / "dir" / "generated.yaml"
    _build_scheduler_manifest(
        modes=["native"],
        baselines=["longrag"],
        datasets=["clerk"],
        top_k=5,
        output_root=tmp_path,
        manifest_path=manifest_path,
    )
    assert manifest_path.exists()


# =====================================================================
# Preflight via mocked helper
# =====================================================================


def test_preflight_as_stage_returns_stagestate(tmp_path, monkeypatch):
    """_preflight_as_stage should return a StageState with done=True if preflight passed."""
    from orchestration.run_all import _preflight_as_stage

    # Build minimal manifest that will pass with mocked env
    manifest = {
        "preflight": {
            "check_env": ["TEST_PREFLIGHT_ENV"],
            "check_models": [],
            "check_datasets": {"required_ids": []},
        }
    }
    monkeypatch.setenv("TEST_PREFLIGHT_ENV", "fake")

    state = RunAllState(started_at="2026-11-04T00:00:00")
    st = _preflight_as_stage(manifest, state)
    assert isinstance(st, StageState)
    # done=True (preflight passes)
    assert st.done is True
    assert st.error is None