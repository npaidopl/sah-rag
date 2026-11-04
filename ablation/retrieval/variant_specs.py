"""Declarative spec for each ablation variant. The runner reads this."""

VARIANT_SPECS = {
    # ============ TIER 1 — structural (universal) ============
    "A4_full_sah": {
        "tier": 1,
        "chunk_source": "A4_full_sah",
        "embed_source": "A4_full_sah",
        "dense": True, "sparse": True, "bm25": False,
        "fusion": "rrf", "rrf_k": 60,
        "aggregation": "best_plus_margin", "margin": 0.05,
    },
    "A7_paragraph_parents": {
        "tier": 1,
        "chunk_source": "A7_paragraph_parents",
        "embed_source": "A7_paragraph_parents",
        "dense": True, "sparse": True, "bm25": False,
        "fusion": "rrf", "rrf_k": 60,
        "aggregation": "best_plus_margin", "margin": 0.05,
    },
    # ============ TIER 1 — structural (structure-rich only) ============
    "A1_flat_hybrid": {
        "tier": 1, "restrict_to_structure_rich": True,
        "chunk_source": "A1_flat_hybrid",
        "embed_source": "A1_flat_hybrid",
        "dense": True, "sparse": True, "bm25": False,
        "fusion": "rrf", "rrf_k": 60,
        "aggregation": "best_plus_margin", "margin": 0.05,
    },
    "A6_boundaries_disabled": {
        "tier": 1, "restrict_to_structure_rich": True,
        "chunk_source": "A6_boundaries_disabled",
        "embed_source": "A6_boundaries_disabled",
        "dense": True, "sparse": True, "bm25": False,
        "fusion": "rrf", "rrf_k": 60,
        "aggregation": "best_plus_margin", "margin": 0.05,
    },
    "A9_placebo_boundaries": {
        "tier": 1, "restrict_to_structure_rich": True,
        "chunk_source": "A9_placebo_boundaries",
        "embed_source": "A9_placebo_boundaries",
        "dense": True, "sparse": True, "bm25": False,
        "fusion": "rrf", "rrf_k": 60,
        "aggregation": "best_plus_margin", "margin": 0.05,
    },
    # ============ TIER 2 — signal ablation (reuses A4) ============
    "A2_structure_dense_only": {
        "tier": 2, "chunk_source": "A4_full_sah", "embed_source": "A4_full_sah",
        "dense": True, "sparse": False, "bm25": False,
        "fusion": "dense_only",
        "aggregation": "best_plus_margin", "margin": 0.05,
    },
    "A3_structure_sparse_only": {
        "tier": 2, "chunk_source": "A4_full_sah", "embed_source": "A4_full_sah",
        "dense": False, "sparse": True, "bm25": False,
        "fusion": "sparse_only",
        "aggregation": "best_plus_margin", "margin": 0.05,
    },
    # ============ TIER 2 — aggregation (reuses A4) ============
    "B1_best_only": {
        "tier": 2, "chunk_source": "A4_full_sah", "embed_source": "A4_full_sah",
        "dense": True, "sparse": True, "bm25": False,
        "fusion": "rrf", "rrf_k": 60,
        "aggregation": "best_only",
    },
    "B3_margin_010": {
        "tier": 2, "chunk_source": "A4_full_sah", "embed_source": "A4_full_sah",
        "dense": True, "sparse": True, "bm25": False,
        "fusion": "rrf", "rrf_k": 60,
        "aggregation": "best_plus_margin", "margin": 0.10,
    },
    "B4_margin_020": {
        "tier": 2, "chunk_source": "A4_full_sah", "embed_source": "A4_full_sah",
        "dense": True, "sparse": True, "bm25": False,
        "fusion": "rrf", "rrf_k": 60,
        "aggregation": "best_plus_margin", "margin": 0.20,
    },
    "B5_sum": {
        "tier": 2, "chunk_source": "A4_full_sah", "embed_source": "A4_full_sah",
        "dense": True, "sparse": True, "bm25": False,
        "fusion": "rrf", "rrf_k": 60,
        "aggregation": "sum",
    },
    "B6_mean": {
        "tier": 2, "chunk_source": "A4_full_sah", "embed_source": "A4_full_sah",
        "dense": True, "sparse": True, "bm25": False,
        "fusion": "rrf", "rrf_k": 60,
        "aggregation": "mean",
    },
    "B7_size_controlled": {
        "tier": 2, "chunk_source": "A4_full_sah", "embed_source": "A4_full_sah",
        "dense": True, "sparse": True, "bm25": False,
        "fusion": "rrf", "rrf_k": 60,
        "aggregation": "size_controlled", "margin": 0.05,
    },
    # ============ TIER 2 — fusion (reuses A4) ============
    "G1_rrf_k10": {
        "tier": 2, "chunk_source": "A4_full_sah", "embed_source": "A4_full_sah",
        "dense": True, "sparse": True, "bm25": False,
        "fusion": "rrf", "rrf_k": 10,
        "aggregation": "best_plus_margin", "margin": 0.05,
    },
    "G3_rrf_k100": {
        "tier": 2, "chunk_source": "A4_full_sah", "embed_source": "A4_full_sah",
        "dense": True, "sparse": True, "bm25": False,
        "fusion": "rrf", "rrf_k": 100,
        "aggregation": "best_plus_margin", "margin": 0.05,
    },
    "G4_weighted_5050": {
        "tier": 2, "chunk_source": "A4_full_sah", "embed_source": "A4_full_sah",
        "dense": True, "sparse": True, "bm25": False,
        "fusion": "weighted", "w_dense": 0.5, "w_sparse": 0.5,
        "aggregation": "best_plus_margin", "margin": 0.05,
    },
    "G5_weighted_7030": {
        "tier": 2, "chunk_source": "A4_full_sah", "embed_source": "A4_full_sah",
        "dense": True, "sparse": True, "bm25": False,
        "fusion": "weighted", "w_dense": 0.7, "w_sparse": 0.3,
        "aggregation": "best_plus_margin", "margin": 0.05,
    },
    "G6_normalized_score": {
        "tier": 2, "chunk_source": "A4_full_sah", "embed_source": "A4_full_sah",
        "dense": True, "sparse": True, "bm25": False,
        "fusion": "normalized",
        "aggregation": "best_plus_margin", "margin": 0.05,
    },
    "G8_random_sparse": {
        "tier": 2, "chunk_source": "A4_full_sah", "embed_source": "A4_full_sah",
        "dense": True, "sparse": True, "bm25": False,
        "fusion": "rrf", "rrf_k": 60,
        "sparse_transform": "randomize",
        "aggregation": "best_plus_margin", "margin": 0.05,
    },
    # ============ TIER 2 — BM25 (reuses A4 dense for D2) ============
    "D2_bge_m3_plus_bm25": {
        "tier": 2, "chunk_source": "A4_full_sah", "embed_source": "A4_full_sah",
        "dense": True, "sparse": False, "bm25": True,
        "fusion": "rrf", "rrf_k": 60,
        "aggregation": "best_plus_margin", "margin": 0.05,
    },
    "D5_bm25_only": {
        "tier": 2, "chunk_source": "A4_full_sah", "embed_source": "A4_full_sah",
        "dense": False, "sparse": False, "bm25": True,
        "fusion": "bm25_only",
        "aggregation": "best_plus_margin", "margin": 0.05,
    },
}