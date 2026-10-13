"""Validate the Step 93 statistical analysis plan."""
from __future__ import annotations

from pathlib import Path

import yaml

PATH = Path("baseline_audit/statistical_analysis_plan.yaml")


def _require(c: bool, m: str) -> None:
    if not c:
        raise SystemExit(m)


def main() -> int:
    d = yaml.safe_load(PATH.read_text(encoding="utf-8"))
    _require(d.get("artifact_type") == "statistical_analysis_plan", "artifact_type changed")
    p = d["plan"]
    _require(p["status"] == "FROZEN_FOR_IMPLEMENTATION", "status must be FROZEN_FOR_IMPLEMENTATION")

    a = d["aggregation"]
    _require(a["per_dataset"]["method"] == "macro_average_over_queries",
             "per_dataset must be macro over queries")
    _require(a["cross_dataset"]["method"] == "macro_average_over_datasets",
             "cross_dataset must be macro over datasets")
    _require(a["global_micro_average"]["reported"] is True, "micro must be reported")
    _require(a["global_micro_average"]["primary"] is False, "micro must not be primary")

    pc = d["pairwise_comparison"]
    _require(pc["pairing_unit"] == "query", "pairing_unit must be query")
    _require(pc["uncertainty"]["method"] == "paired_bootstrap", "must use paired_bootstrap")
    _require(pc["uncertainty"]["resamples"] == 10000, "resamples must be 10000")
    _require(pc["multiple_testing"]["correction"] == "HOLM", "must use HOLM correction")

    cl = d["context_length_analysis"]
    _require(cl["bins_frozen_after_inventory_before_quality_results"] is True,
             "bins must be frozen before quality results")
    _require(len(cl["candidate_bins"]) == 5, "must have 5 candidate bins")

    zr = d["zero_relevance_reporting"]
    _require(zr["cuad_zero_evidence_subset"]["separate_report"] is True,
             "CUAD zero subset must be separate")
    _require(zr["cuad_zero_evidence_subset"]["never_mixed_with_recall_metrics"] is True,
             "must not mix zero with recall")

    print("[OK] Statistical analysis plan")
    print("  aggregation: per-dataset macro -> cross-dataset macro")
    print("  uncertainty: paired_bootstrap (10000 resamples)")
    print("  correction: HOLM")
    print("  context bins:", len(cl["candidate_bins"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())