"""Validate the pilot embedding contract."""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

PATH = Path("baseline_audit/pilot_embedding_contract.yaml")
FROZEN = "5617a9f61b028005a4858fdac845db406aefb181"


def main() -> int:
    data = yaml.safe_load(PATH.read_text(encoding="utf-8"))
    c = data["contract"]
    if c["id"] != "infrastructure_pilot_bge_m3_embedding":
        raise SystemExit("contract.id changed")
    if c["version"] != "0.1.0":
        raise SystemExit("contract.version must be 0.1.0")
    if c["status"] != "FROZEN_FOR_BENCHMARK_PLANNING":
        raise SystemExit("contract.status must be FROZEN_FOR_BENCHMARK_PLANNING")

    m = data["model"]
    if m["model_id"] != "BAAI/bge-m3":
        raise SystemExit("model_id must be BAAI/bge-m3")
    rev = m["model_revision"]
    if len(rev) != 40:
        raise SystemExit("model_revision must be a full 40-char SHA")
    if rev != FROZEN:
        raise SystemExit("model_revision changed")
    if m["revision_pinned"] is not True:
        raise SystemExit("revision_pinned must be true")
    if m["license"]["spdx"] != "MIT":
        raise SystemExit("license spdx must be MIT")
    if m["dense_dimension"] != 1024:
        raise SystemExit("dense_dimension must be 1024")
    if m["dense_normalized"] is not True:
        raise SystemExit("dense_normalized must be true")
    if m["pilot_capabilities_enabled"]["multi_vector"] is not False:
        raise SystemExit("multi_vector must be disabled for the pilot")

    ex = data["execution"]
    for k in (
        "embedding_execution_enabled", "index_execution_enabled",
        "retrieval_execution_enabled", "evaluation_execution_enabled",
        "generation_execution_enabled", "hosted_model_calls_allowed",
        "network_access_allowed", "secrets_required",
    ):
        if ex[k] is not False:
            raise SystemExit(f"execution.{k} must be false")
    if ex["cpu_only"] is not True:
        raise SystemExit("execution.cpu_only must be true")

    ca = data["cache"]
    if ca["network_fallback_allowed"] is not False:
        raise SystemExit("cache.network_fallback_allowed must be false")

    print("[OK] Pilot embedding contract")
    print(f"  model: {m['model_id']}")
    print(f"  revision: {rev}")
    print(f"  revision pinned: {m['revision_pinned']}")
    print(f"  license: {m['license']['spdx']}")
    print(f"  dense dimension: {m['dense_dimension']}")
    print(f"  dense normalization: {m['dense_normalized']}")
    print(f"  sparse method: {m['sparse_method']}")
    print(f"  CPU only: {ex['cpu_only']}")
    print(f"  network allowed: {ex['network_access_allowed']}")
    print(f"  embedding execution enabled: {ex['embedding_execution_enabled']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
