"""Validate the Step 93 architecture fidelity matrix."""
from __future__ import annotations

from pathlib import Path

import yaml

PATH = Path("baseline_audit/architecture_fidelity_matrix.yaml")
ARCHS = ("sah_rag","longrag","raptor","hipporag2","lightrag","pathrag",
         "disretrieval","memorag","kag","metadata_rag","hikey")


def _require(c: bool, m: str) -> None:
    if not c:
        raise SystemExit(m)


def main() -> int:
    d = yaml.safe_load(PATH.read_text(encoding="utf-8"))
    _require(d.get("artifact_type") == "architecture_fidelity_matrix", "artifact_type changed")

    sv = d["status_vocabulary"]
    _require("NATIVE_FAITHFUL" in sv["fidelity_status"], "NATIVE_FAITHFUL missing")
    _require("CONTROLLED_ADAPTATION" in sv["fidelity_status"], "CONTROLLED_ADAPTATION missing")
    _require(
        "PAPER_DERIVED_COMPARATOR" in sv["fidelity_status"],
        "PAPER_DERIVED_COMPARATOR missing",
    )
    _require("BOTH_TRACKS" in sv["benchmark_eligibility"], "BOTH_TRACKS missing")
    _require("DEFERRED" in sv["benchmark_eligibility"], "DEFERRED missing")

    a = d["architectures"]
    _require(set(a.keys()) == set(ARCHS), "architecture set changed")
    for name, rec in a.items():
        _require("implementation_status" in rec, f"{name}: missing implementation_status")
        _require("fidelity_status" in rec, f"{name}: missing fidelity_status")
        _require("benchmark_eligibility" in rec, f"{name}: missing benchmark_eligibility")
        _require("controlled_track" in rec, f"{name}: missing controlled_track")
        _require("native_track" in rec, f"{name}: missing native_track")
        ct = rec["controlled_track"]
        nt = rec["native_track"]
        _require(isinstance(ct.get("eligible"), bool), f"{name}: controlled eligible must be bool")
        _require(isinstance(nt.get("eligible"), bool), f"{name}: native eligible must be bool")
        _require("blockers" in ct, f"{name}: controlled blockers missing")
        _require("blockers" in nt, f"{name}: native blockers missing")
        _require("next_readiness_action" in rec, f"{name}: next_readiness_action missing")

    # Spot checks
    _require(a["sah_rag"]["fidelity_status"] == "NATIVE_FAITHFUL", "sah_rag fidelity changed")
    _require(
        a["pathrag"]["fidelity_status"] == "PAPER_DERIVED_COMPARATOR",
        "pathrag fidelity changed",
    )
    _require(a["hikey"]["benchmark_eligibility"] == "DEFERRED", "hikey must be deferred")

    print("[OK] Architecture fidelity matrix")
    print("  architectures:", len(a))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())