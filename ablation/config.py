"""Ablation study configuration: paths, dataset list, tier groupings,
sampling policy."""
from __future__ import annotations
import os
from pathlib import Path


# =====================================================================
# PATHS
# =====================================================================
SAH_RAG_ROOT = Path(__file__).resolve().parent.parent
CANONICAL_SUITE_ROOT = SAH_RAG_ROOT.parent / "canonical-rag-dataset-suite"
CANONICAL_DATA_ROOT = CANONICAL_SUITE_ROOT / "processed" / "canonical"

WORK_ROOT = Path(os.environ.get("SAH_RAG_ABLATION_WORK", r"D:\rag_work\ablation"))
LOGS_DIR       = WORK_ROOT / "logs"
PLANS_DIR      = WORK_ROOT / "plans"
PARSED_DIR     = WORK_ROOT / "parsed"
CHUNKS_DIR     = WORK_ROOT / "chunks"
EMBEDDINGS_DIR = WORK_ROOT / "embeddings"
INDEXES_DIR    = WORK_ROOT / "indexes"
RESULTS_DIR    = WORK_ROOT / "results"
REPORTS_DIR    = WORK_ROOT / "reports"

HF_CACHE_ROOT = Path(os.environ.get("SAH_RAG_HF_CACHE", r"D:\rag_work\hf_cache"))


# =====================================================================
# DATASETS
# =====================================================================
ALL_DATASETS = [
    "2wiki", "clerk", "cuad", "frames", "greek_bar_bench",
    "hotpotqa", "lara", "long_bench_v2", "loong", "m3docvqa",
    "multifieldqa_en", "multihop_rag", "musique", "narrativeqa",
    "qasper", "quality", "ruler",
]

EXCLUDE_FROM_MACRO = {"m3docvqa", "ruler"}

# Structure-richness classification (from phase05b survey)
STRUCTURE_RICH_DATASETS = ["cuad", "lara", "loong", "narrativeqa", "long_bench_v2"]


# =====================================================================
# VARIANT GROUPS
# =====================================================================
TIER1_UNIVERSAL = ["A4_full_sah", "A7_paragraph_parents"]
TIER1_STRUCTURE_ONLY = ["A1_flat_hybrid", "A9_placebo_boundaries", "A6_boundaries_disabled"]

TIER2_VARIANTS = [
    "A2_structure_dense_only",
    "A3_structure_sparse_only",
    "B1_best_only",
    "B3_margin_010",
    "B4_margin_020",
    "B5_sum",
    "B6_mean",
    "B7_size_controlled",
    "G1_rrf_k10",
    "G3_rrf_k100",
    "G4_weighted_5050",
    "G5_weighted_7030",
    "G6_normalized_score",
    "G8_random_sparse",
    "D2_bge_m3_plus_bm25",
    "D5_bm25_only",
]

TIER3_VARIANTS = ["D3_alt_embedder", "D4_alt_dim"]
TIER3_DATASETS = ["clerk", "cuad", "lara", "qasper", "2wiki"]


# =====================================================================
# SAMPLING POLICY — GT-aware (all GT docs always included)
# =====================================================================
# Only the 4 heaviest datasets + ruler are sampled. All others use full
# corpus. See ablation.chunks.builder.sample_docs_gt_aware for the exact
# procedure (all GT docs first, then random fill).
SAMPLING = {
    "long_bench_v2": 50,
    "loong":           50,
    "narrativeqa":     50,
    "frames":          50,
    "ruler":           30,
}


def sample_limit(dataset: str) -> int | None:
    """Return max docs to use, or None for full dataset."""
    return SAMPLING.get(dataset)


def sample_docs_gt_aware(dataset: str, limit: int) -> set[str]:
    """Return a set of doc_ids: all GT docs + random fill up to limit.

    Deterministic (seed=42). If number of GT docs >= limit, takes first
    `limit` GT docs alphabetically.
    """
    from ablation.loaders import (
        iter_canonical_documents_local, iter_canonical_examples_local,
    )
    import random

    gt_ids: set[str] = set()
    for ex in iter_canonical_examples_local(dataset):
        gt_ids.update(ex.ground_truth.relevant_ids)

    all_docs = {d.doc_id for d in iter_canonical_documents_local(dataset)}
    gt_in_corpus = gt_ids & all_docs

    if len(gt_in_corpus) >= limit:
        return set(sorted(gt_in_corpus)[:limit])

    non_gt = sorted(all_docs - gt_in_corpus)
    rng = random.Random(42)
    n_fill = min(limit - len(gt_in_corpus), len(non_gt))
    fill = rng.sample(non_gt, n_fill)
    return gt_in_corpus | set(fill)


# =====================================================================
# HELPERS
# =====================================================================
def ensure_dirs() -> None:
    for d in [LOGS_DIR, PLANS_DIR, PARSED_DIR, CHUNKS_DIR,
              EMBEDDINGS_DIR, INDEXES_DIR, RESULTS_DIR, REPORTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def tier1_jobs():
    jobs = []
    for ds in ALL_DATASETS:
        for v in TIER1_UNIVERSAL:
            jobs.append((v, ds))
        if ds in STRUCTURE_RICH_DATASETS:
            for v in TIER1_STRUCTURE_ONLY:
                jobs.append((v, ds))
    return jobs


def tier3_jobs():
    return [(v, ds) for v in TIER3_VARIANTS for ds in TIER3_DATASETS]