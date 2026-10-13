"""Validate the pilot parser+chunker contract."""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

CONTRACT = Path("baseline_audit/pilot_parser_chunker_contract.yaml")


def main() -> int:
    data = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    c = data["contract"]
    if c["id"] != "infrastructure_pilot_parser_chunker":
        raise SystemExit("contract.id changed")
    if c["version"] != "0.1.0":
        raise SystemExit("contract.version must be 0.1.0")
    if c["status"] != "APPROVED_FOR_BOUNDED_EXECUTION":
        raise SystemExit("contract.status must be APPROVED_FOR_BOUNDED_EXECUTION")

    ex = data["execution"]
    if ex["parser_enabled"] is not True or ex["chunker_enabled"] is not True:
        raise SystemExit("parser and chunker must be enabled")
    for k in (
        "embedding_enabled", "index_enabled", "retrieval_enabled",
        "evaluation_enabled", "generation_enabled",
        "network_access_allowed", "hosted_requests_allowed",
        "foreign_code_execution_allowed",
    ):
        if ex[k] is not False:
            raise SystemExit(f"execution.{k} must be false")

    rc = data["source_reference_chunks"]
    if rc["role"] != "validation_reference_only":
        raise SystemExit("source_reference_chunks.role changed")
    for k in (
        "equality_with_derived_chunks_required",
        "boundary_equality_required",
        "count_equality_required",
    ):
        if rc[k] is not False:
            raise SystemExit(f"source_reference_chunks.{k} must be false")

    ch = data["chunker"]
    if ch["model_input_token_limit"] != 256:
        raise SystemExit("chunker.model_input_token_limit must be 256")
    if ch["content_token_overlap"] != 32:
        raise SystemExit("chunker.content_token_overlap must be 32")

    lim = data["resource_limits"]
    if lim["cpu_only"] is not True:
        raise SystemExit("resource_limits.cpu_only must be true")
    if lim["gpu_allowed"] is not False:
        raise SystemExit("resource_limits.gpu_allowed must be false")
    if lim["maximum_worker_count"] != 1:
        raise SystemExit("resource_limits.maximum_worker_count must be 1")

    print("[OK] Pilot parser+chunker contract")
    print(f"  id: {c['id']}")
    print(f"  version: {c['version']}")
    print("  parser: enabled")
    print("  chunker: enabled")
    print("  embedding/index/retrieval/eval/gen: disabled")
    print("  source chunks role: validation_reference_only")
    print(f"  chunk size: {ch['model_input_token_limit']} / overlap {ch['content_token_overlap']}")
    print("  CPU only, max workers: 1")
    return 0


if __name__ == "__main__":
    sys.exit(main())
