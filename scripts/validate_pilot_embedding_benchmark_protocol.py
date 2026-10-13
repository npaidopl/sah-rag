"""Validate the pilot embedding benchmark protocol."""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

PATH = Path("baseline_audit/pilot_embedding_benchmark_protocol.yaml")


def main() -> int:
    data = yaml.safe_load(PATH.read_text(encoding="utf-8"))
    runs = data["runs"]
    if runs["batch_sizes"] != [1, 2, 4]:
        raise SystemExit("batch_sizes must be [1, 2, 4]")
    if runs["warmup_run_count"] != 1:
        raise SystemExit("warmup_run_count must be 1")
    if runs["measured_run_count_per_batch_size"] != 3:
        raise SystemExit("measured_run_count_per_batch_size must be 3")

    m = data["measurement"]
    if m["include_model_load_time"] is not False:
        raise SystemExit("include_model_load_time must be false")
    if m["include_persistence_time"] is not False:
        raise SystemExit("include_persistence_time must be false")
    if m["include_embedding_time"] is not True:
        raise SystemExit("include_embedding_time must be true")

    env = data["environment"]
    if env["cpu_only"] is not True:
        raise SystemExit("environment.cpu_only must be true")
    if env["worker_count"] != 1:
        raise SystemExit("environment.worker_count must be 1")
    if env["network_disabled"] is not True:
        raise SystemExit("environment.network_disabled must be true")

    interp = data["interpretation"]
    if interp["linear_extrapolation_prohibited_without_token_adjustment"] is not True:
        raise SystemExit("extrapolation guard missing")

    print("[OK] Pilot embedding benchmark protocol")
    print(f"  input chunks: {data['input']['child_chunk_count']}")
    print(f"  batch sizes: {runs['batch_sizes']}")
    print(f"  warmup runs: {runs['warmup_run_count']}")
    print(f"  measured runs per batch: {runs['measured_run_count_per_batch_size']}")
    print(f"  model load excluded: {not m['include_model_load_time']}")
    print(f"  persistence excluded: {not m['include_persistence_time']}")
    print(f"  publishable: {data['benchmark']['publishable']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
