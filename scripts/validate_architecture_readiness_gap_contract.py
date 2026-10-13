"""Validate the Step 94 readiness gap contract."""
from __future__ import annotations

from pathlib import Path

import yaml

PATH = Path("baseline_audit/architecture_readiness_gap_contract.yaml")

ARCHS = ("rag_system","longrag","raptor","hipporag2","lightrag","pathrag",
         "disretrieval","memorag","kag","metadata_rag","hikey")


def _require(c: bool, m: str) -> None:
    if not c:
        raise SystemExit(m)


def main() -> int:
    d = yaml.safe_load(PATH.read_text(encoding="utf-8"))
    _require(d.get("artifact_type") == "architecture_readiness_gap_contract",
             "artifact_type changed")
    c = d["contract"]
    _require(c["id"] == "eleven_architectures_readiness_gap_registry",
             "contract.id changed")
    _require(c["status"] == "APPROVED_FOR_STATIC_GAP_ANALYSIS",
             "contract.status changed")

    a = d["architectures"]
    _require(tuple(a["expected"]) == ARCHS, "architecture set changed")
    _require(a["expected_count"] == 11, "expected_count must be 11")

    sa = d["static_analysis"]
    _require(sa["allowed"] is True, "static_analysis.allowed must be true")
    _require(sa["import_project_modules"] is False,
             "import_project_modules must be false")
    _require(sa["execute_architecture"] is False,
             "execute_architecture must be false")

    req = d["required_record_fields"]
    _require(len(req) >= 20, "required_record_fields must have >= 20 fields")

    sv = d["status_vocabularies"]
    _require("DATASET_READY" in sv["implementation_status"], "DATASET_READY missing")
    _require("PASSED" in sv["readiness_gate"], "PASSED missing")
    _require("PROJECT_OWNED_REFERENCE" in sv["fidelity_class"],
             "PROJECT_OWNED_REFERENCE missing")

    for p in d["prohibited"]:
        _require(p in ("execute_architecture","load_model","call_llm",
                       "create_index","open_qdrant","create_storage",
                       "execute_retrieval","execute_evaluation"),
                 "unexpected prohibited entry: " + p)

    cl = d["claims"]
    _require(cl["readiness_registry_is_execution_result"] is False,
             "readiness_registry_is_execution_result must be false")
    _require(cl["static_gap_analysis_is_synthetic_proof"] is False,
             "static_gap_analysis_is_synthetic_proof must be false")
    _require(cl["publishable_execution"] is False,
             "publishable_execution must be false")

    print("[OK] Architecture readiness gap contract")
    print("  id:", c["id"])
    print("  architectures:", len(ARCHS))
    print("  static only:", sa["import_project_modules"] is False)
    print("  architecture execution:", sa["execute_architecture"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())