"""Validate the Step 93 fair benchmark protocol."""
from __future__ import annotations

from pathlib import Path

import yaml

PATH = Path("baseline_audit/fair_benchmark_protocol.yaml")

ARCHS = ("sah_rag","longrag","raptor","hipporag2","lightrag","pathrag",
         "disretrieval","memorag","kag","metadata_rag","hikey")
DATASETS = ("ruler","loong","long_bench_v2","lara","greek_bar_bench","cuad","clerk")


def _require(c: bool, m: str) -> None:
    if not c:
        raise SystemExit(m)


def main() -> int:
    d = yaml.safe_load(PATH.read_text(encoding="utf-8"))
    _require(d.get("artifact_type") == "fair_benchmark_protocol", "artifact_type changed")
    p = d["protocol"]
    _require(p["id"] == "wave1_fair_benchmark", "protocol.id changed")
    _require(p["status"] == "FROZEN_FOR_IMPLEMENTATION", "status must be FROZEN_FOR_IMPLEMENTATION")

    s = d["scope"]
    _require(s["generation_enabled"] is False, "generation must be disabled")
    _require(s["retrieval_only"] is True, "retrieval_only must be true")
    _require(s["answer_evaluation_enabled"] is False, "answer eval must be disabled")
    _require(s["llm_as_judge_enabled"] is False, "llm judge must be disabled")

    _require(tuple(d["architectures"]["expected"]) == ARCHS, "architectures set changed")
    _require(d["architectures"]["expected_count"] == 11, "must have 11 architectures")
    _require(tuple(d["datasets"]["expected"]) == DATASETS, "datasets set changed")
    _require(d["datasets"]["expected_count"] == 7, "must have 7 datasets")

    t = d["tracks"]
    ctrl = t["controlled_architectural_track"]
    _require(ctrl["common_components"]["embedder"] == "BAAI/bge-m3", "embedder must be BAAI/bge-m3")
    _require(
        ctrl["common_components"]["generation"] == "disabled",
        "generation must be disabled in track A",
    )
    nat = t["native_fidelity_reference_track"]
    _require(nat["report_separately"] is True, "native track must report separately")
    _require(nat["merge_with_controlled_scores"] is False, "must not merge scores")

    r = d["reporting_rules"]
    _require(r["mixing_tracks_in_single_leaderboard"] is False, "must not mix tracks")
    _require(r["mandatory_track_label"] is True, "must label track")

    print("[OK] Fair benchmark protocol")
    print("  architectures:", len(ARCHS))
    print("  datasets:", len(DATASETS))
    print("  controlled_track:", ctrl["id"])
    print("  native_track:", nat["id"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())