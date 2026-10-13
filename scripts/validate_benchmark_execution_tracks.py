"""Validate the Step 93 benchmark execution tracks."""
from __future__ import annotations

from pathlib import Path

import yaml

PATH = Path("baseline_audit/benchmark_execution_tracks.yaml")


def _require(c: bool, m: str) -> None:
    if not c:
        raise SystemExit(m)


def main() -> int:
    d = yaml.safe_load(PATH.read_text(encoding="utf-8"))
    _require(d.get("artifact_type") == "benchmark_execution_tracks", "artifact_type changed")

    tracks = d["tracks"]
    _require("track_a_controlled_architectural" in tracks, "track_a missing")
    _require("track_b_native_fidelity" in tracks, "track_b missing")
    _require("track_c_deferred_appendix" in tracks, "track_c missing")

    a = tracks["track_a_controlled_architectural"]
    _require(a["role"] == "PRIMARY", "track A must be PRIMARY")
    _require(a["leaderboard_eligible"] is True, "track A must be leaderboard-eligible")

    b = tracks["track_b_native_fidelity"]
    _require(b["role"] == "SECONDARY", "track B must be SECONDARY")
    _require(b["leaderboard_eligible"] is False, "track B must not be leaderboard-eligible")
    _require(
        b["merge_with_controlled_scores"] is False,
        "track B must not merge with A",
    )
    _require(
        b["currently_eligible_count"]
        == len(b["currently_eligible"]),
        "track B currently_eligible_count must match currently_eligible list",
    )
    _require(
        b["potential_candidate_count"]
        == len(b["potential_candidates"]),
        "track B potential_candidate_count must match potential_candidates list",
    )
    overlap = set(b["currently_eligible"]) & set(b["potential_candidates"])
    _require(
        not overlap,
        "track B currently_eligible and potential_candidates must not overlap",
    )

    c = tracks["track_c_deferred_appendix"]
    _require(
        c["numeric_leaderboard_eligible"] is False,
        "track C must not be leaderboard-eligible",
    )

    rules = d["reporting_rules"]
    _require(
        rules["mixing_tracks_in_single_leaderboard"] is False,
        "must not mix tracks",
    )

    print("[OK] Benchmark execution tracks")
    print("  track A eligible:", len(a["eligible_architectures"]))
    print("  track B currently eligible:", b["currently_eligible_count"])
    print("  track B potential candidates:", b["potential_candidate_count"])
    print("  track C (deferred):", len(c["eligible_architectures"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())