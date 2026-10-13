"""Plan (preview only) the infrastructure-only pilot.

Reads the pilot contract, portfolio, environment matrix, readiness
registry, and host profile. Emits deterministic preview experiment
and plan IDs. Does NOT create output directories, materialize the
dataset, or execute anything.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import yaml

CONTRACT = Path("baseline_audit/infrastructure_pilot_contract.yaml")
MATRIX = Path("baseline_audit/architecture_environment_matrix.yaml")
PORTFOLIO = Path("baseline_audit/architecture_portfolio_dispositions.yaml")
HOST = Path("baseline_audit/execution_host_profile.yaml")

PLACEHOLDER_CHECKSUM = "PLANNING_PLACEHOLDER_NOT_MATERIALIZED"


def _canonical_sha256(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _preview_cell(
    *,
    pilot_id: str,
    dataset_id: str,
    dataset_revision: str,
    split: str,
    architecture_id: str,
    adapter_id: str,
    adapter_version: str,
    architecture_configuration_hash: str,
    random_seed: int,
    project_code_revision: str,
    host_profile_id: str,
    pilot_contract_version: str,
) -> str:
    payload = {
        "kind": "pilot_preview_cell",
        "pilot_id": pilot_id,
        "dataset_id": dataset_id,
        "dataset_revision": dataset_revision,
        "split": split,
        "architecture_id": architecture_id,
        "adapter_id": adapter_id,
        "adapter_version": adapter_version,
        "architecture_configuration_hash": architecture_configuration_hash,
        "canonical_input_manifest_sha256": PLACEHOLDER_CHECKSUM,
        "random_seed": random_seed,
        "project_code_revision": project_code_revision,
        "host_profile_id": host_profile_id,
        "pilot_contract_version": pilot_contract_version,
    }
    return "pilot-preview-cell-" + _canonical_sha256(payload)


def _preview_plan(
    *,
    preview_cell_id: str,
    host_profile_id: str,
    pilot_contract_version: str,
) -> str:
    payload = {
        "kind": "pilot_preview_plan",
        "preview_cell_id": preview_cell_id,
        "host_profile_id": host_profile_id,
        "pilot_contract_version": pilot_contract_version,
    }
    return "pilot-preview-plan-" + _canonical_sha256(payload)


def main() -> int:
    contract = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    matrix = yaml.safe_load(MATRIX.read_text(encoding="utf-8"))
    portfolio = yaml.safe_load(PORTFOLIO.read_text(encoding="utf-8"))
    host = yaml.safe_load(HOST.read_text(encoding="utf-8"))

    pilot_id = "infrastructure-pilot-wave1-cpu-v1"
    dataset = contract["dataset"]
    planning = contract["planning"]
    contract_version = contract["contract"]["version"]
    host_profile_id = host["profile"]["profile_id"]

    preview_cells: dict[str, str] = {}
    preview_plans: dict[str, str] = {}
    for arch in planning["architecture_order"]:
        if portfolio["architectures"][arch]["operational_status"] != "READY_FOR_DATA":
            raise SystemExit(f"{arch}: not READY_FOR_DATA in portfolio")
        if matrix["architectures"][arch]["pilot_eligible"] is not True:
            raise SystemExit(f"{arch}: not pilot_eligible in matrix")

        cell = _preview_cell(
            pilot_id=pilot_id,
            dataset_id=dataset["dataset_id"],
            dataset_revision=dataset["dataset_revision"],
            split=dataset["split"],
            architecture_id=arch,
            adapter_id=f"{arch}-adapter",
            adapter_version="0.0.0-planning",
            architecture_configuration_hash=f"{arch}-cfg-planning",
            random_seed=planning["random_seed"],
            project_code_revision="planning",
            host_profile_id=host_profile_id,
            pilot_contract_version=contract_version,
        )
        preview_cells[arch] = cell
        preview_plans[arch] = _preview_plan(
            preview_cell_id=cell,
            host_profile_id=host_profile_id,
            pilot_contract_version=contract_version,
        )

    print("[OK] Infrastructure pilot plan preview")
    print(f"  pilot id: {pilot_id}")
    print(f"  execution role: {contract['scope']['execution_role']}")
    print(f"  architectures: {len(preview_cells)}")
    print("  dataset materialized: False")
    print("  executable experiment cells created: False")
    print("  output directories created: False")
    print()
    for arch in planning["architecture_order"]:
        env_status = matrix["architectures"][arch]["environment_status"]
        print(f"  {arch}:")
        print(f"    preview cell: {preview_cells[arch]}")
        print(f"    preview plan: {preview_plans[arch]}")
        print(f"    environment: {env_status}")
    total_stages = len(preview_cells) * len(planning["required_stages"])
    print()
    print(f"  total planned stages: {total_stages}")
    print("  architecture execution performed: False")
    print("  model execution performed: False")
    print("  evaluation execution performed: False")
    print("  generation performed: False")
    print("  publishable execution enabled: False")
    return 0


if __name__ == "__main__":
    sys.exit(main())
