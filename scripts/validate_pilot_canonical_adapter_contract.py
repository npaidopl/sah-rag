"""Validate the pilot canonical adapter contract."""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

CONTRACT = Path("baseline_audit/pilot_canonical_adapter_contract.yaml")


def main() -> int:
    if not CONTRACT.is_file():
        raise SystemExit(f"missing contract: {CONTRACT}")
    data = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))

    c = data["contract"]
    if c["id"] != "pilot_to_rag_system_canonical":
        raise SystemExit("contract.id changed")
    if c["version"] != "0.1.0":
        raise SystemExit("contract.version must be 0.1.0")
    if c["status"] != "FROZEN_FOR_MATERIALIZATION":
        raise SystemExit("contract.status must be FROZEN_FOR_MATERIALIZATION")

    src = data["source"]
    if src["dataset_id"] != "infrastructure-pilot-synthetic-v1":
        raise SystemExit("source.dataset_id changed")
    if src["immutable"] is not True:
        raise SystemExit("source must be immutable")

    tgt = data["target"]
    if tgt["adapter_id"] != "pilot_to_rag_system_canonical":
        raise SystemExit("target.adapter_id changed")
    if "documents.jsonl" not in tgt["required_files"]:
        raise SystemExit("target must include documents.jsonl")
    if "examples.jsonl" not in tgt["required_files"]:
        raise SystemExit("target must include examples.jsonl")

    mp = data["mapping"]
    if mp["document_count"] != 12:
        raise SystemExit("mapping.document_count must be 12")
    if mp["example_count"] != 6:
        raise SystemExit("mapping.example_count must be 6")
    if mp["source_chunks"]["consumed_as_authoritative_chunks"] is not False:
        raise SystemExit("source chunks must not be authoritative")
    if mp["source_chunks"]["validation_reference_only"] is not True:
        raise SystemExit("source chunks must be validation reference only")
    if mp["reference_answers"]["created"] is not False:
        raise SystemExit("reference_answers.created must be false")
    if mp["reference_answers"]["permitted"] is not False:
        raise SystemExit("reference_answers.permitted must be false")

    safety = data["safety"]
    for k in (
        "baseline_execution_enabled",
        "parser_execution_enabled",
        "chunker_execution_enabled",
        "embedding_execution_enabled",
        "index_execution_enabled",
        "retrieval_execution_enabled",
        "evaluation_execution_enabled",
        "generation_enabled",
        "network_required",
    ):
        if safety[k] is not False:
            raise SystemExit(f"safety.{k} must be false")

    print("[OK] Pilot canonical adapter contract")
    print(f"  id: {c['id']}")
    print(f"  version: {c['version']}")
    print(f"  source dataset: {src['dataset_id']}")
    print(f"  target dataset: {tgt['dataset_id']}")
    print(f"  document count: {mp['document_count']}")
    print(f"  example count: {mp['example_count']}")
    authoritative = mp['source_chunks']['consumed_as_authoritative_chunks']
    print(f"  source chunks authoritative: {authoritative}")
    print(f"  reference answers created: {mp['reference_answers']['created']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
