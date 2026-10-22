"""Tests for the local import + enrich stage."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = (
    REPO_ROOT / "scripts" / "import_metadata_rag_colab_embeddings.py"
)


def _load_module() -> Any:
    sys.path.insert(0, str(REPO_ROOT / "src"))
    spec = importlib.util.spec_from_file_location(
        "import_metadata_rag_colab_embeddings", SCRIPT_PATH
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


imp = _load_module()


# --- Pure helpers ---------------------------------------------------------

def test_constants() -> None:
    assert imp.IMPORT_ARTIFACT_TYPE == "metadata_rag_embeddings_imported"
    assert imp.IMPORT_ARTIFACT_VERSION == "1.0"
    assert imp.DENSE_DIMENSION == 1024
    assert imp.LOCAL_CONFIG_SNAPSHOT_MARKER.startswith("colab://")


def test_sha256_file_roundtrip(tmp_path: Path) -> None:
    p = tmp_path / "x.bin"
    p.write_bytes(b"hello")
    assert imp._sha256_file(p) == hashlib.sha256(b"hello").hexdigest()


def test_write_jsonl_deterministic(tmp_path: Path) -> None:
    p = tmp_path / "x.jsonl"
    n = imp._write_jsonl(p, [{"b": 2, "a": 1}, {"c": 3}])
    assert n == 2
    lines = p.read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[0]) == {"a": 1, "b": 2}


def test_to_tuple_vector_rejects_short() -> None:
    with pytest.raises(imp.ImportError_, match="length"):
        imp._to_tuple_vector([0.1, 0.2])


def test_to_tuple_vector_rejects_non_list() -> None:
    with pytest.raises(imp.ImportError_, match="array"):
        imp._to_tuple_vector("not-a-list")


def test_build_local_config_is_deterministic() -> None:
    a = imp._build_local_config(
        model_id="BAAI/bge-m3",
        model_revision="5617a9f61b028005a4858fdac845db406aefb181",
        device="cuda",
        use_fp16=True,
    )
    b = imp._build_local_config(
        model_id="BAAI/bge-m3",
        model_revision="5617a9f61b028005a4858fdac845db406aefb181",
        device="cuda",
        use_fp16=True,
    )
    assert a.config_hash() == b.config_hash()
    assert a.local_snapshot_path == imp.LOCAL_CONFIG_SNAPSHOT_MARKER


# --- Enrichment -----------------------------------------------------------

def _unit_vector(dim: int, index: int) -> list[float]:
    vec = [0.0] * dim
    vec[index] = 1.0
    return vec


def _fake_state() -> Any:
    """Build a minimal MetadataRAGState from the synthetic fixture."""
    from baselines.metadata_rag_adapter import (
        build_indexed_documents,
    )
    from baselines.metadata_rag_state import build_state
    from sah_rag.canonical.loaders import iter_validated_models
    from sah_rag.canonical.models import CanonicalDocument

    fixture = (
        REPO_ROOT
        / "processed"
        / "canonical"
        / "infrastructure-pilot-synthetic-v1"
    )
    docs = tuple(
        iter_validated_models(
            fixture / "documents.jsonl", CanonicalDocument
        )
    )
    indexed = build_indexed_documents(
        docs, dataset_id="fixture"
    )
    return build_state(indexed, dataset_id="fixture")


def test_enrich_document_embeddings_happy_path() -> None:
    state = _fake_state()
    config = imp._build_local_config(
        model_id="BAAI/bge-m3",
        model_revision="5617a9f61b028005a4858fdac845db406aefb181",
        device="cpu",
        use_fp16=False,
    )
    rows = [
        {
            "doc_id": d.doc_id,
            "dense_vector": _unit_vector(1024, i % 1024),
        }
        for i, d in enumerate(state.documents)
    ]
    out = imp._enrich_document_embeddings(
        state=state, encoded_rows=rows, config=config
    )
    assert len(out) == len(state.documents)
    for rec, d in zip(out, state.documents, strict=True):
        assert rec.doc_id == d.doc_id
        assert rec.metadata_record_id == d.metadata_record_id
        assert rec.dense_dimension == 1024
        assert rec.dense_normalized is True


def test_enrich_rejects_missing_embedding() -> None:
    state = _fake_state()
    config = imp._build_local_config(
        model_id="BAAI/bge-m3",
        model_revision="5617a9f61b028005a4858fdac845db406aefb181",
        device="cpu",
        use_fp16=False,
    )
    rows = [
        {
            "doc_id": d.doc_id,
            "dense_vector": _unit_vector(1024, 0),
        }
        for d in state.documents[:-1]
    ]
    with pytest.raises(imp.ImportError_, match="bijection"):
        imp._enrich_document_embeddings(
            state=state, encoded_rows=rows, config=config
        )


def test_enrich_rejects_extra_embedding() -> None:
    state = _fake_state()
    config = imp._build_local_config(
        model_id="BAAI/bge-m3",
        model_revision="5617a9f61b028005a4858fdac845db406aefb181",
        device="cpu",
        use_fp16=False,
    )
    rows = [
        {
            "doc_id": d.doc_id,
            "dense_vector": _unit_vector(1024, 0),
        }
        for d in state.documents
    ]
    rows.append(
        {"doc_id": "orphan", "dense_vector": _unit_vector(1024, 1)}
    )
    with pytest.raises(imp.ImportError_, match="bijection"):
        imp._enrich_document_embeddings(
            state=state, encoded_rows=rows, config=config
        )


def test_enrich_query_embeddings_happy_path() -> None:
    config = imp._build_local_config(
        model_id="BAAI/bge-m3",
        model_revision="5617a9f61b028005a4858fdac845db406aefb181",
        device="cpu",
        use_fp16=False,
    )
    rows = [
        {"query_id": "q-1", "dense_vector": _unit_vector(1024, 0)},
        {"query_id": "q-2", "dense_vector": _unit_vector(1024, 1)},
    ]
    out = imp._enrich_query_embeddings(
        dataset_id="fixture", encoded_rows=rows, config=config
    )
    assert len(out) == 2
    assert out[0].query_id == "q-1"
    assert out[1].query_id == "q-2"
    # The persisted record must not contain the raw query text.
    dumped = out[0].model_dump()
    assert "query_text" not in dumped
    assert "query_text_sha256" in dumped


def test_enrich_query_rejects_duplicate() -> None:
    config = imp._build_local_config(
        model_id="BAAI/bge-m3",
        model_revision="5617a9f61b028005a4858fdac845db406aefb181",
        device="cpu",
        use_fp16=False,
    )
    rows = [
        {"query_id": "q-1", "dense_vector": _unit_vector(1024, 0)},
        {"query_id": "q-1", "dense_vector": _unit_vector(1024, 1)},
    ]
    with pytest.raises(imp.ImportError_, match="duplicate"):
        imp._enrich_query_embeddings(
            dataset_id="fixture", encoded_rows=rows, config=config
        )
