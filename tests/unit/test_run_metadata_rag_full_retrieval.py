"""Tests for the L5 full-retrieval script (pure helpers)."""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = (
    REPO_ROOT / "scripts" / "run_metadata_rag_full_retrieval.py"
)


def _load_module() -> Any:
    sys.path.insert(0, str(REPO_ROOT / "src"))
    spec = importlib.util.spec_from_file_location(
        "run_metadata_rag_full_retrieval", SCRIPT_PATH
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rr = _load_module()


def test_constants() -> None:
    assert rr.RETRIEVAL_ARTIFACT_TYPE == "metadata_rag_retrieval"
    assert rr.RETRIEVAL_ARTIFACT_VERSION == "1.0"
    assert rr.EXPECTED_DATASETS == (
        "clerk",
        "cuad",
        "greek_bar_bench",
        "lara",
    )


def test_sha256_file(tmp_path: Path) -> None:
    p = tmp_path / "x.bin"
    p.write_bytes(b"payload")
    assert rr._sha256_file(p) == hashlib.sha256(b"payload").hexdigest()


def test_write_jsonl_deterministic(tmp_path: Path) -> None:
    p = tmp_path / "x.jsonl"
    n = rr._write_jsonl(p, [{"b": 2, "a": 1}])
    assert n == 1
    assert p.read_text(encoding="utf-8") == '{"a":1,"b":2}\n'


def test_read_jsonl_skips_blanks(tmp_path: Path) -> None:
    p = tmp_path / "x.jsonl"
    p.write_text('{"a":1}\n\n{"b":2}\n', encoding="utf-8")
    assert len(rr._read_jsonl(p)) == 2


def test_load_query_texts_rejects_duplicates(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "colab_bundle"
    bundle_dir.mkdir()
    (bundle_dir / "queries.jsonl").write_text(
        '{"query_id":"q-1","text":"a"}\n'
        '{"query_id":"q-1","text":"b"}\n',
        encoding="utf-8",
    )
    with pytest.raises(
        rr.MetadataRAGRetrievalRunError, match="duplicate"
    ):
        rr._load_query_texts(tmp_path)


def test_verify_bijections_happy_path() -> None:
    from baselines.metadata_rag_adapter import (
        build_indexed_documents,
    )
    from baselines.metadata_rag_embedder import (
        EncoderConfig,
        build_document_embedding,
        build_query_embedding,
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
    indexed = build_indexed_documents(docs, dataset_id="fixture")
    state = build_state(indexed, dataset_id="fixture")

    cfg = EncoderConfig(local_snapshot_path="C:/fake")
    one = tuple([1.0] + [0.0] * 1023)
    doc_embs = tuple(
        build_document_embedding(
            config=cfg,
            dataset_id="fixture",
            doc_id=d.doc_id,
            document_order=d.document_order,
            metadata_record_id=d.metadata_record_id,
            source_text=d.text_normalized,
            dense_vector=one,
        )
        for d in state.documents
    )
    q_embs = tuple(
        build_query_embedding(
            config=cfg,
            dataset_id="fixture",
            query_id=f"q-{i}",
            query_text=f"query {i}",
            dense_vector=one,
        )
        for i in range(3)
    )
    texts = {f"q-{i}": f"query text {i}" for i in range(3)}
    # Must not raise.
    rr._verify_bijections(
        state=state,
        document_embeddings=doc_embs,
        query_embeddings=q_embs,
        query_texts=texts,
    )


def test_verify_bijections_rejects_missing_doc() -> None:
    from baselines.metadata_rag_adapter import (
        build_indexed_documents,
    )
    from baselines.metadata_rag_embedder import (
        EncoderConfig,
        build_document_embedding,
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
    indexed = build_indexed_documents(docs, dataset_id="fixture")
    state = build_state(indexed, dataset_id="fixture")

    cfg = EncoderConfig(local_snapshot_path="C:/fake")
    one = tuple([1.0] + [0.0] * 1023)
    doc_embs = tuple(
        build_document_embedding(
            config=cfg,
            dataset_id="fixture",
            doc_id=d.doc_id,
            document_order=d.document_order,
            metadata_record_id=d.metadata_record_id,
            source_text=d.text_normalized,
            dense_vector=one,
        )
        for d in state.documents[:-1]
    )
    with pytest.raises(
        rr.MetadataRAGRetrievalRunError, match="bijection"
    ):
        rr._verify_bijections(
            state=state,
            document_embeddings=doc_embs,
            query_embeddings=(),
            query_texts={},
        )
