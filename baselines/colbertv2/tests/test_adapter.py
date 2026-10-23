"""Offline tests for the ColBERTv2 baseline adapter.

Two tiers:

1. Always run (fast, offline, CPU, no model):
   - identity
   - config parsing
   - controlled-mode rejection
   - chunking correctness
   - path isolation under work_dir
   All of these complete in < 1 s total.

2. Heavy (skipped unless colbert-ai + checkpoint + CUDA are all present):
   - full build_index + retrieve
   The skipif gate enforces the "no network, <10 s" rule from the
   operating manual on CI and CPU-only dev machines.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from baselines.base import BaselineConfig, BaselineStatus, RunMode
from sah_rag.canonical.models import CanonicalDocument
from baselines.colbertv2.adapter import ColBERTv2Baseline


# ---------------------------------------------------------------------------
# Availability gates
# ---------------------------------------------------------------------------

def _colbert_available() -> bool:
    """True iff colbert-ai is importable in this env."""
    return importlib.util.find_spec("colbert") is not None


def _checkpoint_cached() -> bool:
    """True iff colbert-ir/colbertv2.0 is present in the HF cache."""
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    return (cache_dir / "models--colbert-ir--colbertv2.0").exists()


def _gpu_available() -> bool:
    """True iff torch is importable AND CUDA is present."""
    try:
        import torch  # local import; not top-level
        return bool(torch.cuda.is_available())
    except Exception:
        return False


_HEAVY_OK = _colbert_available() and _checkpoint_cached() and _gpu_available()
_HEAVY_REASON = (
    "heavy test requires colbert-ai + cached checkpoint + CUDA; "
    f"colbert={_colbert_available()}, "
    f"cached={_checkpoint_cached()}, "
    f"cuda={_gpu_available()}"
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make_doc(doc_id: str, text: str) -> CanonicalDocument:
    return CanonicalDocument(
        doc_id=doc_id,
        dataset="test",
        text_raw=text,
        text_normalized=text,
    )


def _native_cfg(top_k: int = 3, **extra_overrides) -> BaselineConfig:
    extra = {
        "checkpoint": "colbert-ir/colbertv2.0",
        "doc_maxlen": 512,
        "nbits": 2,
        "ncells": 1,
        "chunk_size": 1800,
        "chunk_overlap": 200,
        "experiment": "colbertv2_native",
        "index_name": "corpus",
    }
    extra.update(extra_overrides)
    return BaselineConfig(
        name="colbertv2", mode=RunMode.NATIVE, top_k=top_k, extra=extra
    )


# ---------------------------------------------------------------------------
# Tier 1 — always run
# ---------------------------------------------------------------------------

def test_identity() -> None:
    """Contract: identity attributes are set as frozen class attrs."""
    assert ColBERTv2Baseline.name == "colbertv2"
    assert ColBERTv2Baseline.version == "0.1.0"
    assert ColBERTv2Baseline.status == BaselineStatus.EXECUTABLE
    assert ColBERTv2Baseline.supports_native is True
    assert ColBERTv2Baseline.supports_controlled is False
    assert ColBERTv2Baseline.is_llm_based is False


def test_native_instantiation(tmp_path: Path) -> None:
    """Native mode instantiates with embedder=None, llm=None."""
    baseline = ColBERTv2Baseline(
        config=_native_cfg(), work_dir=tmp_path / "idx"
    )
    assert (tmp_path / "idx").is_dir()
    assert baseline.config.mode == RunMode.NATIVE
    assert baseline._built is False


def test_controlled_mode_rejected(tmp_path: Path) -> None:
    """Controlled mode is rejected because supports_controlled=False."""
    cfg = BaselineConfig(
        name="colbertv2", mode=RunMode.CONTROLLED, top_k=3, extra={}
    )
    with pytest.raises(ValueError, match="controlled"):
        ColBERTv2Baseline(
            config=cfg, work_dir=tmp_path / "idx", embedder=object()
        )


def test_config_rejects_overlap_ge_chunk_size(tmp_path: Path) -> None:
    """Sanity check on our own config validation."""
    cfg = _native_cfg(chunk_size=200, chunk_overlap=200)
    with pytest.raises(ValueError, match="chunk_overlap"):
        ColBERTv2Baseline(config=cfg, work_dir=tmp_path / "idx")


def test_chunk_text_empty() -> None:
    assert ColBERTv2Baseline._chunk_text("", 1800, 200) == []
    assert ColBERTv2Baseline._chunk_text("    \n\t", 1800, 200) == []


def test_chunk_text_short() -> None:
    text = "Short text."
    assert ColBERTv2Baseline._chunk_text(text, 1800, 200) == [text]


def test_chunk_text_long_multiple_of_step() -> None:
    # chunk_size=1800, overlap=200 -> step=1600.
    # len=5000: i=0,1600,3200 -> 3 chunks, all full 1800.
    text = "X" * 5000
    chunks = ColBERTv2Baseline._chunk_text(text, 1800, 200)
    assert len(chunks) == 3
    assert all(len(c) == 1800 for c in chunks)
    # Overlap invariant: last `overlap` chars of chunk n == first `overlap`
    # chars of chunk n+1.
    assert chunks[0][-200:] == chunks[1][:200]
    assert chunks[1][-200:] == chunks[2][:200]


def test_chunk_text_long_partial_tail() -> None:
    # len=4000, chunk_size=1800, overlap=200 -> step=1600.
    # i=0 (full), i=1600 (full), i=3200 (tail 800).
    text = "X" * 4000
    chunks = ColBERTv2Baseline._chunk_text(text, 1800, 200)
    assert [len(c) for c in chunks] == [1800, 1800, 800]


def test_paths_are_under_work_dir(tmp_path: Path) -> None:
    """No hardcoded paths: all persistence points live under work_dir."""
    wd = tmp_path / "idx"
    baseline = ColBERTv2Baseline(config=_native_cfg(), work_dir=wd)
    assert str(baseline._collection_path).startswith(str(wd))
    assert str(baseline._mapping_path).startswith(str(wd))
    assert str(baseline._experiments_root).startswith(str(wd))


def test_retrieve_before_build_raises(tmp_path: Path) -> None:
    baseline = ColBERTv2Baseline(
        config=_native_cfg(), work_dir=tmp_path / "idx"
    )
    with pytest.raises(RuntimeError, match="build_index"):
        baseline.retrieve("any query")


def test_retrieve_empty_query_returns_empty(tmp_path: Path) -> None:
    baseline = ColBERTv2Baseline(
        config=_native_cfg(), work_dir=tmp_path / "idx"
    )
    # Mark as built so retrieve() reaches the query check instead of the
    # build-order check. No model is loaded for an empty query.
    baseline._built = True
    assert baseline.retrieve("") == []
    assert baseline.retrieve("   ") == []


def test_sanitize_tsv_replaces_whitespace() -> None:
    """Tabs and newlines must not break the TSV row format."""
    raw = "a\tb\nc\r\nd"
    out = ColBERTv2Baseline._sanitize_for_tsv(raw)
    assert "\t" not in out
    assert "\n" not in out
    assert "\r" not in out


# ---------------------------------------------------------------------------
# Tier 2 — heavy (skipped on CPU/offline)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _HEAVY_OK, reason=_HEAVY_REASON)
def test_build_and_retrieve_end_to_end(tmp_path: Path) -> None:
    """Full native pipeline on a tiny corpus (GPU + cached ckpt only)."""
    docs = [
        _make_doc("d1", "The cat sat on the mat."),
        _make_doc("d2", "Dogs chase balls in the park."),
        _make_doc("d3", "RAG systems retrieve relevant context."),
        _make_doc("d4", "Long-context documents need special handling."),
        _make_doc("d5", "Legal contracts are structured hierarchically."),
    ]
    baseline = ColBERTv2Baseline(
        config=_native_cfg(top_k=3), work_dir=tmp_path / "idx"
    )
    baseline.build_index(docs)

    # Artifacts exist and mapping is well-formed.
    assert baseline._collection_path.exists()
    assert baseline._mapping_path.exists()
    mapping = json.loads(baseline._mapping_path.read_text(encoding="utf-8"))
    assert len(mapping) == len(docs)  # each doc is one short chunk
    for entry in mapping.values():
        assert {"chunk_id", "doc_id", "text"} <= entry.keys()

    # Retrieval contract.
    results = baseline.retrieve("RAG retrieval", top_k=3)
    assert isinstance(results, list)
    assert 0 < len(results) <= 3
    assert [r.rank for r in results] == list(range(1, len(results) + 1))
    corpus_ids = {d.doc_id for d in docs}
    for r in results:
        assert r.doc_id in corpus_ids
        assert r.text
        assert r.chunk_id.startswith(r.doc_id + "_c")

    baseline.cleanup()