"""Print the current embedding plan without executing anything."""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

CONTRACT = Path("baseline_audit/pilot_embedding_contract.yaml")
MANIFEST = Path("baseline_audit/pilot_embedding_manifest.yaml")
BENCH = Path("baseline_audit/pilot_embedding_benchmark_protocol.yaml")
QD = Path("baseline_audit/pilot_qdrant_collection_plan.yaml")
COL = Path("baseline_audit/pilot_colab_transfer_protocol.yaml")


def main() -> int:
    for path in (CONTRACT, MANIFEST, BENCH, QD, COL):
        if not path.is_file():
            print(f"missing: {path}", file=sys.stderr)
            return 1

    contract = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    manifest = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    bench = yaml.safe_load(BENCH.read_text(encoding="utf-8"))
    qd = yaml.safe_load(QD.read_text(encoding="utf-8"))

    print("[OK] Pilot embedding plan")
    print(f"  contract: {contract['contract']['id']} v{contract['contract']['version']}")
    print(f"  model: {contract['model']['model_id']}")
    print(f"  revision: {contract['model']['model_revision']}")
    print(f"  cache root: {contract['cache']['root']}")
    print(f"  cache_status: {manifest['model']['cache_status']}")
    print(f"  dense dimension: {contract['model']['dense_dimension']}")
    print(f"  sparse method: {contract['model']['sparse_method']}")
    print(f"  input chunks expected: {contract['source']['expected_child_chunk_count']}")
    print(f"  batch sizes: {bench['runs']['batch_sizes']}")
    print(f"  warmup runs: {bench['runs']['warmup_run_count']}")
    print(f"  measured runs per batch: {bench['runs']['measured_run_count_per_batch_size']}")
    print(f"  qdrant backend: {qd['backend']['type']} {qd['backend']['mode']}")
    print(f"  qdrant creation approved: {qd['safety']['collection_creation_approved']}")
    print(f"  embedding execution approved: {manifest['approval']['embedding_execution_approved']}")
    print(f"  network allowed: {contract['execution']['network_access_allowed']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
