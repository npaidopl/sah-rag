"""Tests for the Metadata-RAG evaluation script."""

from __future__ import annotations
# --- SAH-RAG path resolution (sanitized from local absolute paths) ---
import os as _os
from pathlib import Path as _Path
_SAH_RAG_REPO_ROOT = _Path(__file__).resolve().parents[1]
_SAH_RAG_PROJECT_ROOT = _os.environ.get(
    "SAH_RAG_PROJECT_ROOT",
    str(_SAH_RAG_REPO_ROOT.parent / "Project 1"),
)
_SAH_RAG_CANONICAL_ROOT = _os.environ.get(
    "SAH_RAG_CANONICAL_ROOT",
    str(_Path(_SAH_RAG_PROJECT_ROOT) / "processed" / "canonical"),
)
# ---------------------------------------------------------------------

import importlib.util
import math
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = (
    REPO_ROOT / "scripts" / "run_metadata_rag_evaluation.py"
)


def _load_module() -> Any:
    sys.path.insert(0, str(REPO_ROOT / "src"))
    spec = importlib.util.spec_from_file_location(
        "run_metadata_rag_evaluation", SCRIPT_PATH
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ev = _load_module()


# --- Constants ------------------------------------------------------------

def test_constants() -> None:
    assert ev.EXPECTED_DATASETS == (
        "clerk",
        "cuad",
        "greek_bar_bench",
        "lara",
    )
    assert ev.CUTOFFS == (1, 5, 10, 20, 50, 100)
    assert ev.PRIMARY_METRIC == "macro_document_recall_at_10"


# --- Hit@k ----------------------------------------------------------------

def test_hit_at_k_positive() -> None:
    ranked = ["a", "b", "c", "d"]
    relevant = {"c"}
    assert ev._hit_at_k(ranked, relevant, 3) == 1.0


def test_hit_at_k_negative() -> None:
    ranked = ["a", "b", "c"]
    relevant = {"z"}
    assert ev._hit_at_k(ranked, relevant, 3) == 0.0


# --- Recall@k -------------------------------------------------------------

def test_recall_at_k_exact() -> None:
    ranked = ["a", "b", "c", "d"]
    relevant = {"a", "c", "z"}
    assert math.isclose(
        ev._recall_at_k(ranked, relevant, 4), 2 / 3, abs_tol=1e-12
    )


def test_recall_at_k_zero_relevant_rejected() -> None:
    with pytest.raises(ev.MetadataRAGEvaluationError):
        ev._recall_at_k(["a"], set(), 1)


# --- Precision@k ----------------------------------------------------------

def test_precision_at_k_fixed_cutoff() -> None:
    ranked = ["a", "b"]
    relevant = {"a"}
    # 1 relevant in top 5 => 1/5, not 1/2.
    assert math.isclose(
        ev._precision_at_k(ranked, relevant, 5), 0.2, abs_tol=1e-12
    )


# --- MRR@k ----------------------------------------------------------------

def test_mrr_first_at_1() -> None:
    assert ev._reciprocal_rank(["a", "b"], {"a"}, 5) == 1.0


def test_mrr_first_at_3() -> None:
    assert math.isclose(
        ev._reciprocal_rank(["a", "b", "c"], {"c"}, 5),
        1 / 3,
        abs_tol=1e-12,
    )


def test_mrr_zero_when_no_relevant_in_cutoff() -> None:
    assert ev._reciprocal_rank(["a", "b"], {"z"}, 2) == 0.0


# --- nDCG@k ---------------------------------------------------------------

def test_ndcg_perfect_ranking() -> None:
    # Both relevant at positions 1 and 2.
    assert math.isclose(
        ev._ndcg_at_k(["a", "b"], {"a", "b"}, 2), 1.0, abs_tol=1e-12
    )


def test_ndcg_penalizes_late_relevant() -> None:
    # Relevant at position 3.
    val = ev._ndcg_at_k(["x", "y", "a"], {"a"}, 3)
    assert 0.0 < val < 1.0


def test_ndcg_zero_relevant_rejected() -> None:
    with pytest.raises(ev.MetadataRAGEvaluationError):
        ev._ndcg_at_k(["a"], set(), 1)


# --- Query metrics + aggregation -----------------------------------------

def test_compute_query_metrics_shape() -> None:
    entry = ev._compute_query_metrics(
        query_id="q-1",
        ranked_doc_ids=["a", "b", "c"],
        relevant_doc_ids={"a"},
    )
    assert entry["query_id"] == "q-1"
    for k in ev.CUTOFFS:
        for name in (
            f"hit_at_{k}",
            f"precision_at_{k}",
            f"recall_at_{k}",
            f"mrr_at_{k}",
            f"ndcg_at_{k}",
        ):
            assert name in entry["metrics"]


def test_aggregate_macro_mean() -> None:
    per_query = [
        {
            "query_id": "q1",
            "metrics": {"recall_at_10": 1.0, "hit_at_10": 1.0},
        },
        {
            "query_id": "q2",
            "metrics": {"recall_at_10": 0.5, "hit_at_10": 1.0},
        },
    ]
    agg = ev._aggregate_macro(per_query)
    assert agg["query_count"] == 2
    assert agg["mean"]["recall_at_10"] == 0.75
    assert agg["mean"]["hit_at_10"] == 1.0


def test_aggregate_macro_rejects_empty() -> None:
    with pytest.raises(ev.MetadataRAGEvaluationError):
        ev._aggregate_macro([])


# --- GT + retrieval loading ----------------------------------------------

def test_load_retrieval_results_dedupes_by_canonical(
    tmp_path: Path,
) -> None:
    import json as _json

    retrieval_dir = tmp_path / "retrieval-v1"
    retrieval_dir.mkdir(parents=True)
    # Three ranked items: two windows of d1 (w0, w1) then one of d2.
    # After dedup the sequence must be [d1, d2].
    results_path = retrieval_dir / "results.jsonl"
    row = {
        "query_id": "q-1",
        "ranked_items": [
            {"canonical_doc_id": "d1", "rank": 1, "score": 0.9},
            {"canonical_doc_id": "d1", "rank": 2, "score": 0.8},
            {"canonical_doc_id": "d2", "rank": 3, "score": 0.7},
        ],
    }
    results_path.write_text(
        _json.dumps(row) + "\n", encoding="utf-8"
    )
    loaded = ev._load_retrieval_results(tmp_path)
    assert loaded["q-1"] == ["d1", "d2"]


def test_load_retrieval_results_dedupes_interleaved(
    tmp_path: Path,
) -> None:
    import json as _json

    retrieval_dir = tmp_path / "retrieval-v1"
    retrieval_dir.mkdir(parents=True)
    row = {
        "query_id": "q-1",
        "ranked_items": [
            {"canonical_doc_id": "a", "rank": 1, "score": 0.9},
            {"canonical_doc_id": "b", "rank": 2, "score": 0.8},
            {"canonical_doc_id": "a", "rank": 3, "score": 0.7},
            {"canonical_doc_id": "c", "rank": 4, "score": 0.6},
            {"canonical_doc_id": "b", "rank": 5, "score": 0.5},
        ],
    }
    (retrieval_dir / "results.jsonl").write_text(
        _json.dumps(row) + "\n", encoding="utf-8"
    )
    loaded = ev._load_retrieval_results(tmp_path)
    assert loaded["q-1"] == ["a", "b", "c"]


def test_load_ground_truth_from_fixture() -> None:
    # Use one of the real datasets but stop after a few entries.
    # The four real datasets are present under Project 1; check via
    # the repo-local synthetic fallback if the real one is absent.
    real_canonical = Path(
        _SAH_RAG_CANONICAL_ROOT
    )
    if not (real_canonical / "clerk").is_dir():
        pytest.skip("real canonical datasets not present on this host")
    gt = ev._load_ground_truth("clerk")
    assert len(gt) > 0
    for qid, relevant in gt.items():
        assert isinstance(qid, str)
        assert isinstance(relevant, tuple)
