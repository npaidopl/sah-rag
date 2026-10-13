"""Validate the Step 93 precision equivalence protocol."""
from __future__ import annotations

from pathlib import Path

import yaml

PATH = Path("baseline_audit/precision_equivalence_protocol.yaml")


def _require(c: bool, m: str) -> None:
    if not c:
        raise SystemExit(m)


def main() -> int:
    d = yaml.safe_load(PATH.read_text(encoding="utf-8"))
    _require(d.get("artifact_type") == "precision_equivalence_protocol", "artifact_type changed")
    p = d["protocol"]
    _require(p["id"] == "fp32_vs_fp16_equivalence", "protocol.id changed")
    _require(p["status"] == "FROZEN_FOR_IMPLEMENTATION", "status must be FROZEN_FOR_IMPLEMENTATION")

    stages = [s["id"] for s in d["stages"]]
    _require("FP32_REFERENCE" in stages, "FP32_REFERENCE missing")
    _require("FP16_CANDIDATE" in stages, "FP16_CANDIDATE missing")
    _require("RETRIEVAL_EQUIVALENCE" in stages, "RETRIEVAL_EQUIVALENCE missing")

    t = d["acceptance_thresholds"]
    _require(t["median_same_record_cosine_minimum"] == 0.999, "cosine threshold must be 0.999")
    _require(t["mean_top_10_overlap_minimum"] == 0.99, "top10 overlap must be 0.99")
    _require(t["absolute_recall_at_10_delta_maximum"] == 0.005, "recall delta must be 0.005")

    dr = d["decision_rule"]
    _require(dr["if_pass"]["production_precision"] == "FP16", "if_pass must be FP16")
    _require(dr["if_fail"]["production_precision"] == "FP32", "if_fail must be FP32")

    print("[OK] Precision equivalence protocol")
    print("  stages:", len(stages))
    print("  if_pass precision: FP16")
    print("  if_fail precision: FP32")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())