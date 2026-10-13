"""Deterministic planner for the Step 93 fair benchmark protocol.

Reads the 8 YAML contracts, cross-checks counts, and updates the
fair_benchmark_manifest.yaml with actual summary values. Never
executes any pipeline stage or loads any model.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]

PROTOCOL = REPO_ROOT / "baseline_audit" / "fair_benchmark_protocol.yaml"
FIDELITY = REPO_ROOT / "baseline_audit" / "architecture_fidelity_matrix.yaml"
SUBSTITUTION = REPO_ROOT / "baseline_audit" / "component_substitution_policy.yaml"
TRACKS = REPO_ROOT / "baseline_audit" / "benchmark_execution_tracks.yaml"
TASK_MATRIX = REPO_ROOT / "baseline_audit" / "dataset_retrieval_task_matrix.yaml"
PRECISION = REPO_ROOT / "baseline_audit" / "precision_equivalence_protocol.yaml"
STATS = REPO_ROOT / "baseline_audit" / "statistical_analysis_plan.yaml"
MANIFEST = REPO_ROOT / "baseline_audit" / "fair_benchmark_manifest.yaml"


def _require(c: bool, m: str) -> None:
    if not c:
        raise SystemExit("[FAIL] " + m)


def _load(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-update-manifest", action="store_true")
    args = parser.parse_args()

    _ = _load(PROTOCOL)
    fidelity = _load(FIDELITY)
    substitution = _load(SUBSTITUTION)
    _ = _load(TRACKS)
    task_matrix = _load(TASK_MATRIX)
    precision = _load(PRECISION)
    stats = _load(STATS)
    manifest = _load(MANIFEST)

    # --- Architecture track counts ---
    archs = fidelity["architectures"]
    both = sum(
        1 for a in archs.values()
        if a["benchmark_eligibility"] == "BOTH_TRACKS"
    )
    primary_only = sum(
        1 for a in archs.values()
        if a["benchmark_eligibility"] == "PRIMARY_CONTROLLED_TRACK"
    )
    native_only = sum(
        1 for a in archs.values()
        if a["benchmark_eligibility"] == "SECONDARY_NATIVE_TRACK"
    )
    deferred = sum(
        1 for a in archs.values()
        if a["benchmark_eligibility"] == "DEFERRED"
    )
    excluded = sum(
        1 for a in archs.values()
        if a["benchmark_eligibility"] == "EXCLUDED_WITH_REASON"
    )

    controlled_eligible = sum(
        1 for a in archs.values() if a["controlled_track"]["eligible"]
    )
    native_eligible = sum(
        1 for a in archs.values() if a["native_track"]["eligible"]
    )

    # --- Component counts ---
    comps = substitution["components"]
    arch_defining = sum(1 for c in comps.values() if c["default_role"] == "ARCHITECTURE_DEFINING")
    arch_supporting = sum(
        1 for c in comps.values()
        if c["default_role"] == "ARCHITECTURE_SUPPORTING"
    )
    optional = sum(1 for c in comps.values() if c["default_role"] == "OPTIONAL_DOWNSTREAM")

    # --- LLM stage counts ---
    llm_index = comps["llm_index_time"]["default_role"]
    llm_query = comps["llm_query_time"]["default_role"]
    llm_final = comps["llm_final_generation"]["default_role"]

    # --- Storage counts ---
    storage_policy = substitution["storage_substitution_policy"]

    # --- Dataset task counts ---
    ds = task_matrix["datasets"]
    doc_level = sum(1 for r in ds.values() if r["primary_evaluation_unit"] == "CANONICAL_DOCUMENT")
    span_level = sum(1 for r in ds.values() if r["primary_evaluation_unit"] == "SOURCE_SPAN")
    passage_level = sum(1 for r in ds.values() if r["primary_evaluation_unit"] == "PASSAGE")
    needle_level = sum(
        1 for r in ds.values()
        if r["primary_evaluation_unit"] == "NEEDLE_OR_EVIDENCE_ITEM"
    )
    zero_rel = sum(1 for r in ds.values() if r["zero_relevance_policy"] == "EXPLICIT_NEGATIVE_CASE")

    # --- Precision/stats frozen? ---
    precision_frozen = precision["protocol"]["status"] == "FROZEN_FOR_IMPLEMENTATION"
    stats_frozen = stats["plan"]["status"] == "FROZEN_FOR_IMPLEMENTATION"

    _require(len(archs) == 11, "must have 11 architectures")
    _require(len(ds) == 7, "must have 7 datasets")
    _require(both + primary_only + native_only + deferred + excluded == 11,
             "architecture eligibility counts must sum to 11")

    # --- Update manifest ---
    if not args.no_update_manifest:
        manifest["execution"]["status"] = "SUCCEEDED"
        r = manifest["results"]
        r["architecture_records_created"] = True
        r["controlled_track_count"] = controlled_eligible
        r["native_track_count"] = native_eligible
        r["both_track_count"] = both
        r["deferred_count"] = deferred
        r["architecture_defining_components_classified"] = True
        r["substitutions_classified"] = True
        r["llm_stages_classified"] = True
        r["storage_backends_classified"] = True
        r["dataset_task_records_created"] = True
        r["precision_protocol_frozen"] = precision_frozen
        r["statistical_plan_frozen"] = stats_frozen
        MANIFEST.write_text(
            yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )

    # --- Output ---
    print("[OK] Fair benchmark methodology plan")
    print("  architectures:", len(archs))
    print("  datasets:", len(ds))
    print()
    print("  controlled-track eligible:", controlled_eligible)
    print("  native-track eligible:", native_eligible)
    print("  both-track eligible:", both)
    print("  deferred:", deferred)
    print("  excluded with reason:", excluded)
    print()
    print("  architecture-defining components:", arch_defining)
    print("  architecture-supporting components:", arch_supporting)
    print("  optional downstream components:", optional)
    print()
    print("  LLM index-time role:", llm_index)
    print("  LLM query-time role:", llm_query)
    print("  LLM final-generation role:", llm_final)
    print()
    print("  native storage policies:", len(storage_policy["native_storage_required_when"]))
    print("  Qdrant substitution policies:", len(storage_policy["common_qdrant_allowed_when"]))
    print()
    print("  dataset task mappings:", len(ds))
    print("    document-level:", doc_level)
    print("    span-level:", span_level)
    print("    passage-level:", passage_level)
    print("    needle/evidence-level:", needle_level)
    print("    zero-relevance datasets:", zero_rel)
    print()
    print("  precision equivalence protocol frozen:", precision_frozen)
    print("  statistical analysis plan frozen:", stats_frozen)
    print()
    print("  pipeline executed: False")
    print("  tokenizer loaded: False")
    print("  model loaded: False")
    print("  retrieval executed: False")
    print("  evaluation executed: False")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())