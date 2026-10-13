"""Validate the infrastructure pilot dataset contract."""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

CONTRACT = Path("baseline_audit/infrastructure_pilot_dataset_contract.yaml")


def main() -> int:
    if not CONTRACT.is_file():
        raise SystemExit(f"missing contract: {CONTRACT}")
    data = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    c = data["contract"]
    if c["id"] != "infrastructure_pilot_synthetic_dataset":
        raise SystemExit("contract.id changed")
    if c["version"] != "1.0.0":
        raise SystemExit("contract.version must be 1.0.0")
    if c["status"] != "FROZEN_FOR_MATERIALIZATION":
        raise SystemExit("contract.status must be FROZEN_FOR_MATERIALIZATION")

    ds = data["dataset"]
    if ds["dataset_id"] != "infrastructure-pilot-synthetic-v1":
        raise SystemExit("dataset_id changed")
    if ds["dataset_revision"] != "1.0.0":
        raise SystemExit("dataset_revision must be 1.0.0")
    if ds["real_dataset"] is not False:
        raise SystemExit("real_dataset must be false")
    if ds["publishable_dataset_result"] is not False:
        raise SystemExit("publishable_dataset_result must be false")

    counts = data["counts"]
    if counts["document_count"] != 12:
        raise SystemExit("document_count must be 12")
    if counts["duplicate_content_pair_count"] != 6:
        raise SystemExit("duplicate_content_pair_count must be 6")
    if counts["unique_content_count"] != 6:
        raise SystemExit("unique_content_count must be 6")
    if counts["chunk_count"] != 24:
        raise SystemExit("chunk_count must be 24")
    if counts["query_count"] != 6:
        raise SystemExit("query_count must be 6")
    if counts["judgment_count"] != 12:
        raise SystemExit("judgment_count must be 12")

    safety = data["safety"]
    for k in (
        "baseline_execution_enabled", "model_execution_enabled",
        "retrieval_execution_enabled", "evaluation_execution_enabled",
        "final_generation_enabled", "network_access_required",
        "foreign_code_execution_allowed",
    ):
        if safety[k] is not False:
            raise SystemExit(f"safety.{k} must be false")

    print("[OK] Infrastructure pilot dataset contract")
    print(f"  id: {c['id']}")
    print(f"  version: {c['version']}")
    print(f"  dataset: {ds['dataset_id']} (rev {ds['dataset_revision']})")
    print(f"  documents: {counts['document_count']}")
    print(f"  unique content groups: {counts['unique_content_count']}")
    print(f"  chunks: {counts['chunk_count']}")
    print(f"  queries: {counts['query_count']}")
    print(f"  judgments: {counts['judgment_count']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
