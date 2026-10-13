"""Validate the execution host profile."""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

PROFILE = Path("baseline_audit/execution_host_profile.yaml")


def main() -> int:
    if not PROFILE.is_file():
        raise SystemExit(f"missing host profile: {PROFILE}")
    data = yaml.safe_load(PROFILE.read_text(encoding="utf-8"))

    if data["artifact_type"] != "execution_host_profile":
        raise SystemExit("artifact_type changed")

    p = data["profile"]
    if not str(p["profile_id"]).startswith("execution-host-"):
        raise SystemExit("profile_id must start with execution-host-")

    for key in ("physical_cpu_count", "logical_cpu_count", "system_ram_bytes"):
        if p[key] < 0:
            raise SystemExit(f"{key} must be non-negative")
    if p["physical_cpu_count"] > p["logical_cpu_count"]:
        raise SystemExit("physical_cpu_count must not exceed logical_cpu_count")

    if p["gpu_available"] is False and p["gpu_count"] != 0:
        raise SystemExit("gpu_count must be 0 when gpu_available is false")
    if p["gpu_available"] is False and p["gpu_devices"]:
        raise SystemExit("gpu_devices must be empty when gpu_available is false")
    if p["cuda_runtime_available"] is False and p["cuda_version"] is not None:
        raise SystemExit("cuda_version must be null without CUDA runtime")

    storage = data["storage"]
    if storage["working_directory_free_bytes"] < 0:
        raise SystemExit("free bytes must be non-negative")
    if storage["working_directory_free_bytes"] > storage["working_directory_total_bytes"]:
        raise SystemExit("free bytes must not exceed total")

    policy = data["policy"]
    if policy["maximum_approved_cost"]["amount"] < 0:
        raise SystemExit("approved cost must be non-negative")
    if policy["secrets_available"] is not False:
        raise SystemExit("secrets_available must be false in the frozen profile")

    collection = data["collection"]
    for key in (
        "baseline_execution_performed",
        "model_loaded",
        "dataset_loaded",
        "network_probe_performed",
        "sensitive_identifiers_collected",
    ):
        if collection[key] is not False:
            raise SystemExit(f"collection.{key} must be false")

    print("[OK] Execution host profile")
    print(f"  profile id: {p['profile_id']}")
    print(f"  operating system: {p['operating_system']}")
    print(f"  CPU: {p['cpu_model']}")
    print(f"  logical CPUs: {p['logical_cpu_count']}")
    print(f"  RAM bytes: {p['system_ram_bytes']}")
    print(f"  GPU available: {p['gpu_available']}")
    print(f"  CUDA-capable GPU available: {p['cuda_capable_gpu_available']}")
    print(f"  GPU count: {p['gpu_count']}")
    print(f"  Docker engine reachable: {data['tools']['docker_engine_reachable']}")
    print(f"  network policy: {policy['network_policy']}")
    print(f"  hosted-model policy: {policy['hosted_model_policy']}")
    cost = policy["maximum_approved_cost"]
    print(f"  approved cost: {cost['amount']} {cost['currency']}")
    print("  baseline execution performed: False")
    return 0


if __name__ == "__main__":
    sys.exit(main())
