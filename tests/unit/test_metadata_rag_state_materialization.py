"""Tests for the Metadata-RAG state materialization script helpers."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = (
    REPO_ROOT
    / "scripts"
    / "run_metadata_rag_state_materialization.py"
)


def _load_module() -> Any:
    sys.path.insert(0, str(REPO_ROOT / "src"))
    spec = importlib.util.spec_from_file_location(
        "metadata_rag_state_materialization", SCRIPT_PATH
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mat = _load_module()


def test_default_dataset_set_is_the_four_targets() -> None:
    assert mat.EXPECTED_DATASETS == (
        "clerk",
        "cuad",
        "greek_bar_bench",
        "lara",
    )


def test_bundle_artifact_type_is_stable() -> None:
    assert mat.BUNDLE_ARTIFACT_TYPE == "metadata_rag_colab_bundle"
    assert mat.BUNDLE_ARTIFACT_VERSION == "1.0"


def test_sha256_bytes_roundtrip() -> None:
    payload = b"hello world"
    assert mat._sha256_bytes(payload) == hashlib.sha256(payload).hexdigest()


def test_write_jsonl_atomic_is_deterministic(tmp_path: Path) -> None:
    path = tmp_path / "out.jsonl"
    rows = [
        {"b": 2, "a": 1},
        {"d": 4, "c": 3},
    ]
    n = mat._write_jsonl_atomic(path, rows)
    assert n == 2
    content = path.read_text(encoding="utf-8")
    assert content.endswith("\n")
    lines = content.splitlines()
    assert len(lines) == 2
    # Sorted keys within each row.
    assert json.loads(lines[0]) == {"a": 1, "b": 2}
    assert json.loads(lines[1]) == {"c": 3, "d": 4}


def test_write_json_is_deterministic(tmp_path: Path) -> None:
    path = tmp_path / "x.json"
    mat._write_json(path, {"z": 1, "a": 2})
    content = path.read_text(encoding="utf-8")
    assert content == '{"a":2,"z":1}\n'


def test_write_jsonl_empty_input(tmp_path: Path) -> None:
    path = tmp_path / "empty.jsonl"
    n = mat._write_jsonl_atomic(path, [])
    assert n == 0
    assert path.read_bytes() == b"\n"


def test_package_bundle_roundtrip(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    for name in (
        "manifest.json",
        "windows.jsonl",
        "queries.jsonl",
        "checksums.json",
    ):
        (bundle_dir / name).write_text("{}\n", encoding="utf-8")
    target = tmp_path / "bundle.tar.gz"
    mat._package_bundle(bundle_dir, target)
    assert target.is_file()
    import tarfile

    with tarfile.open(target, "r:gz") as tar:
        names = sorted(tar.getnames())
    assert names == [
        "checksums.json",
        "manifest.json",
        "queries.jsonl",
        "windows.jsonl",
    ]


def test_is_already_done_returns_false_for_missing_dir(
    tmp_path: Path,
) -> None:
    assert mat._is_already_done(tmp_path / "nope") is False


def test_is_already_done_returns_false_for_partial_state(
    tmp_path: Path,
) -> None:
    dataset_dir = tmp_path / "ds"
    (dataset_dir / "state-v1").mkdir(parents=True)
    (dataset_dir / "state-v1" / "manifest.json").write_text(
        json.dumps({"artifact_type": "metadata_rag_state"}),
        encoding="utf-8",
    )
    # No bundle files, no tarball.
    assert mat._is_already_done(dataset_dir) is False
