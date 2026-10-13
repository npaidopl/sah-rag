"""Unit tests for the pilot embedding execution helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER = REPO_ROOT / "scripts" / "run_pilot_embedding_materialization.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("_pilot_mat", RUNNER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_text_sha256_is_stable():
    runner = _load_runner()
    a = runner.text_sha256("hello world")
    b = runner.text_sha256("hello world")
    assert a == b
    assert len(a) == 64


def test_normalize_sparse_sorts_and_drops_nonpositive():
    runner = _load_runner()
    idx, vals = runner.normalize_sparse({"3": 0.5, "1": 0.2, "2": 0.0, "5": 0.9})
    assert idx == (1, 3, 5)
    assert vals == (0.2, 0.5, 0.9)


def test_normalize_sparse_empty():
    runner = _load_runner()
    idx, vals = runner.normalize_sparse({})
    assert idx == ()
    assert vals == ()


def test_determinism_check_reports_byte_identical():
    runner = _load_runner()

    class FakeSparse:
        def __init__(self, indices, values):
            self.indices = tuple(indices)
            self.values = tuple(values)

    class FakeRecord:
        def __init__(self):
            self.dense_vector = (0.1, 0.2, 0.3)
            self.sparse_vector = FakeSparse((0, 2), (0.5, 0.9))

    result = runner.determinism_check([FakeRecord()], [FakeRecord()])
    assert result["status"] == "BYTE_IDENTICAL"
    assert result["max_dense_absolute_difference"] == 0.0


def test_determinism_check_reports_numerically_stable():
    runner = _load_runner()

    class FakeSparse:
        def __init__(self, indices, values):
            self.indices = tuple(indices)
            self.values = tuple(values)

    class FakeRecord:
        def __init__(self, dv, sv):
            self.dense_vector = dv
            self.sparse_vector = sv

    a = FakeRecord((1.0, 2.0), FakeSparse((0,), (0.5,)))
    b = FakeRecord((1.0 + 1e-9, 2.0), FakeSparse((0,), (0.5 + 1e-9,)))
    result = runner.determinism_check([a], [b])
    assert result["status"] == "NUMERICALLY_STABLE_NOT_BYTE_IDENTICAL"
    assert result["max_dense_absolute_difference"] <= 1e-6


def test_determinism_check_reports_unstable():
    runner = _load_runner()

    class FakeSparse:
        def __init__(self, indices, values):
            self.indices = tuple(indices)
            self.values = tuple(values)

    class FakeRecord:
        def __init__(self, dv, sv):
            self.dense_vector = dv
            self.sparse_vector = sv

    a = FakeRecord((1.0,), FakeSparse((0,), (0.5,)))
    b = FakeRecord((1.0 + 1e-3,), FakeSparse((0,), (0.5,)))
    result = runner.determinism_check([a], [b])
    assert result["status"] == "UNSTABLE"


def test_contract_hash_is_deterministic():
    runner = _load_runner()
    import yaml
    contract = yaml.safe_load(
        (REPO_ROOT / "baseline_audit" / "pilot_embedding_execution_contract.yaml")
        .read_text(encoding="utf-8")
    )
    a = runner.compute_embedding_config_hash(contract)
    b = runner.compute_embedding_config_hash(contract)
    assert a == b
    assert len(a) == 64


def test_contract_hash_excludes_mutable_env_versions():
    runner = _load_runner()
    import copy

    import yaml

    contract_path = REPO_ROOT / "baseline_audit" / "pilot_embedding_execution_contract.yaml"
    base = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
    mutated = copy.deepcopy(base)
    mutated["model"]["environment_versions"]["python"] = "9.99.99"
    assert (
        runner.compute_embedding_config_hash(base)
        == runner.compute_embedding_config_hash(mutated)
    )