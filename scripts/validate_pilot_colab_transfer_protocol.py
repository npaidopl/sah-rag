"""Validate the pilot Colab transfer protocol."""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

PATH = Path("baseline_audit/pilot_colab_transfer_protocol.yaml")


def main() -> int:
    data = yaml.safe_load(PATH.read_text(encoding="utf-8"))
    t = data["transfer"]
    prohibited = set(t["prohibited"])
    for required in ("api keys", "access tokens", "model weights", "generated answers"):
        if required not in prohibited:
            raise SystemExit(f"prohibited must contain: {required}")

    ex = data["execution"]
    if ex["approved"] is not False:
        raise SystemExit("execution.approved must be false")
    if ex["fallback_role"] != "remote_gpu_embedding_only":
        raise SystemExit("fallback_role must be remote_gpu_embedding_only")
    if ex["retrieval_on_colab_allowed"] is not False:
        raise SystemExit("retrieval_on_colab_allowed must be false")
    if ex["qdrant_on_colab_allowed"] is not False:
        raise SystemExit("qdrant_on_colab_allowed must be false")

    print("[OK] Pilot Colab transfer protocol")
    print(f"  protocol id: {data['protocol']['id']}")
    print(f"  status: {data['protocol']['status']}")
    print(f"  transfer approved: {ex['approved']}")
    print(f"  fallback role: {ex['fallback_role']}")
    print(f"  retrieval on colab: {ex['retrieval_on_colab_allowed']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
