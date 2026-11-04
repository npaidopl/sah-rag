"""Build the run plan (list of jobs) from config + variant specs."""
from ablation.config import (
    ALL_DATASETS, STRUCTURE_RICH_DATASETS,
    TIER2_VARIANTS, TIER3_VARIANTS, TIER3_DATASETS,
)
from ablation.retrieval.variant_specs import VARIANT_SPECS


def _is_tier1(name: str) -> bool:
    return VARIANT_SPECS[name]["tier"] == 1


def build_plan() -> list[dict]:
    jobs = []
    # Tier 1
    for vname, spec in VARIANT_SPECS.items():
        if not _is_tier1(vname):
            continue
        if spec.get("restrict_to_structure_rich"):
            datasets = STRUCTURE_RICH_DATASETS
        else:
            datasets = ALL_DATASETS
        for ds in datasets:
            jobs.append({"variant": vname, "dataset": ds, "tier": 1})
    # Tier 2
    for vname in TIER2_VARIANTS:
        if vname not in VARIANT_SPECS:
            raise KeyError(f"TIER2 variant '{vname}' has no spec")
        for ds in ALL_DATASETS:
            jobs.append({"variant": vname, "dataset": ds, "tier": 2})
    # Tier 3 (placeholder — depends on alt embedder)
    for vname in TIER3_VARIANTS:
        for ds in TIER3_DATASETS:
            jobs.append({"variant": vname, "dataset": ds, "tier": 3,
                         "status": "placeholder"})
    return jobs


def plan_summary(jobs: list[dict]) -> dict:
    from collections import Counter
    by_tier = Counter(j["tier"] for j in jobs)
    by_variant = Counter(j["variant"] for j in jobs)
    return {"total": len(jobs), "by_tier": dict(by_tier), "by_variant": dict(by_variant)}