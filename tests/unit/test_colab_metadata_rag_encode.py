"""Tests for the Colab encoding script (pure helpers + dry-run)."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = (
    REPO_ROOT / "scripts" / "colab_metadata_rag_encode.py"
)


def _load_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "colab_metadata_rag_encode", SCRIPT_PATH
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


enc = _load_module()


# --- Constants ------------------------------------------------------------

def test_default_model_constants() -> None:
    assert enc.MODEL_ID_DEFAULT == "BAAI/bge-m3"
    assert (
        enc.MODEL_REVISION_DEFAULT
        == "5617a9f61b028005a4858fdac845db406aefb181"
    )
    assert enc.DENSE_DIMENSION == 1024


def test_artifact_constants() -> None:
    assert enc.ENCODING_ARTIFACT_TYPE == "metadata_rag_embeddings"
    assert enc.ENCODING_ARTIFACT_VERSION == "1.0"


# --- Deterministic I/O -----------------------------------------------------

def test_sha256_bytes_roundtrip() -> None:
    payload = b"hello world"
    assert enc._sha256_bytes(payload) == hashlib.sha256(
        payload
    ).hexdigest()


def test_write_json_sorted_deterministic(tmp_path: Path) -> None:
    path = tmp_path / "x.json"
    enc._write_json(path, {"z": 1, "a": 2})
    assert path.read_text(encoding="utf-8") == '{"a":2,"z":1}\n'


def test_write_jsonl_atomic_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "x.jsonl"
    rows = [{"b": 2, "a": 1}, {"c": 3}]
    n = enc._write_jsonl_atomic(path, rows)
    assert n == 2
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0]) == {"a": 1, "b": 2}


def test_read_jsonl_skips_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "x.jsonl"
    path.write_text('{"a":1}\n\n{"b":2}\n', encoding="utf-8")
    rows = enc._read_jsonl(path)
    assert len(rows) == 2


# --- Contract encoder ------------------------------------------------------

def test_contract_encoder_shape() -> None:
    c = enc._ContractEncoder()
    out = c.encode(["hello", "world"], batch_size=2, max_length=10)
    assert len(out) == 2
    for vec in out:
        assert len(vec) == 1024


def test_contract_encoder_is_unit_norm() -> None:
    c = enc._ContractEncoder()
    (vec,) = c.encode(["hello"], batch_size=1, max_length=10)
    norm = math.sqrt(sum(x * x for x in vec))
    assert abs(norm - 1.0) < 1e-9


def test_contract_encoder_is_deterministic() -> None:
    c = enc._ContractEncoder()
    a = c.encode(["hello"], batch_size=1, max_length=10)
    b = c.encode(["hello"], batch_size=1, max_length=10)
    assert a == b


# --- Batch encoding wrapper -----------------------------------------------

def test_encode_in_batches_empty_returns_empty() -> None:
    c = enc._ContractEncoder()
    rows = enc._encode_in_batches(
        c, [], batch_size=2, max_length=10, label="x"
    )
    assert rows == []


def test_encode_in_batches_full_length() -> None:
    c = enc._ContractEncoder()
    rows = enc._encode_in_batches(
        c, ["a", "b", "c", "d", "e"], batch_size=2,
        max_length=10, label="x",
    )
    assert len(rows) == 5


def test_encode_in_batches_rejects_bad_batch_size() -> None:
    c = enc._ContractEncoder()
    with pytest.raises(enc.ColabEncodingError, match="batch_size"):
        enc._encode_in_batches(
            c, ["a"], batch_size=0, max_length=10, label="x"
        )


def test_encode_in_batches_rejects_bad_max_length() -> None:
    c = enc._ContractEncoder()
    with pytest.raises(enc.ColabEncodingError, match="max_length"):
        enc._encode_in_batches(
            c, ["a"], batch_size=1, max_length=1, label="x"
        )


# --- End-to-end dry-run on synthetic bundle -------------------------------

def _write_synthetic_bundle(bundle_dir: Path) -> Path:
    (bundle_dir).mkdir(parents=True, exist_ok=True)
    windows_path = bundle_dir / "windows.jsonl"
    queries_path = bundle_dir / "queries.jsonl"
    enc._write_jsonl_atomic(
        windows_path,
        [
            {"doc_id": "d-1", "text": "The Amber Observatory."},
            {"doc_id": "d-2", "text": "The Cedar Archive."},
        ],
    )
    enc._write_jsonl_atomic(
        queries_path,
        [
            {"query_id": "q-1", "text": "Where is the Amber Observatory?"},
        ],
    )
    manifest_path = bundle_dir / "manifest.json"
    enc._write_json(
        manifest_path,
        {
            "artifact_type": "metadata_rag_colab_bundle",
            "artifact_version": "1.0",
            "dataset_id": "fixture",
            "window_count": 2,
            "query_count": 1,
        },
    )
    checksums = {
        "windows.jsonl": enc._sha256_file(windows_path),
        "queries.jsonl": enc._sha256_file(queries_path),
        "manifest.json": enc._sha256_file(manifest_path),
    }
    enc._write_json(bundle_dir / "checksums.json", checksums)

    tar_path = bundle_dir.parent / "bundle.tar.gz"
    with __import__("tarfile").open(tar_path, "w:gz") as tar:
        for name in (
            "manifest.json",
            "windows.jsonl",
            "queries.jsonl",
            "checksums.json",
        ):
            tar.add(bundle_dir / name, arcname=name)
    return tar_path


def test_dry_run_end_to_end(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "bundle_src"
    bundle_tar = _write_synthetic_bundle(bundle_dir)
    output_tar = tmp_path / "out.tar.gz"
    work_dir = tmp_path / "work"

    manifest = enc.run_encode(
        bundle_tar=bundle_tar,
        output_tar=output_tar,
        work_dir=work_dir,
        dry_run=True,
        model_id="BAAI/bge-m3",
        model_revision="5617a9f61b028005a4858fdac845db406aefb181",
        device="cpu",
        use_fp16=False,
        snapshot_dir=None,
        window_batch_size=2,
        query_batch_size=2,
        window_max_length=8192,
        query_max_length=4096,
    )
    assert manifest["encoder_kind"] == "DETERMINISTIC_CONTRACT_ENCODER"
    assert manifest["window_count"] == 2
    assert manifest["query_count"] == 1
    assert output_tar.is_file()

    import tarfile as _tar

    with _tar.open(output_tar, "r:gz") as tar:
        names = sorted(tar.getnames())
    assert names == [
        "checksums.json",
        "manifest.json",
        "query_embeddings.jsonl",
        "window_embeddings.jsonl",
    ]


def test_extract_bundle_rejects_bad_checksum(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "b"
    bundle_tar = _write_synthetic_bundle(bundle_dir)
    # Tamper with the tar by rebuilding with a wrong checksums entry.
    tmp_dir = tmp_path / "tampered"
    tmp_dir.mkdir()
    with __import__("tarfile").open(bundle_tar, "r:gz") as tar:
        tar.extractall(tmp_dir, filter="data")
    (tmp_dir / "windows.jsonl").write_text(
        '{"doc_id":"evil","text":"x"}\n', encoding="utf-8"
    )
    bad_tar = tmp_path / "bad.tar.gz"
    with __import__("tarfile").open(bad_tar, "w:gz") as tar:
        for name in (
            "manifest.json",
            "windows.jsonl",
            "queries.jsonl",
            "checksums.json",
        ):
            tar.add(tmp_dir / name, arcname=name)

    with pytest.raises(
        enc.ColabEncodingError, match="checksum mismatch"
    ):
        enc._extract_bundle(bad_tar, tmp_path / "out")
