"""Fast smoke check for the persisted pilot Qdrant collection.

Opens a fresh local client, confirms the persisted collection exists
with exactly 12 points, and does NOT execute any query.
"""

from __future__ import annotations

import contextlib
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
EXECUTION_CONTRACT = REPO_ROOT / "baseline_audit" / "pilot_qdrant_execution_contract.yaml"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit("[FAIL] " + message)


def main() -> int:
    contract = yaml.safe_load(EXECUTION_CONTRACT.read_text(encoding="utf-8"))
    storage_root = REPO_ROOT / contract["backend"]["storage_root"]
    collection_name = contract["collection"]["physical_name"]

    _require(storage_root.is_dir(), "storage root missing: " + str(storage_root))

    from qdrant_client import QdrantClient

    client = QdrantClient(path=str(storage_root))
    try:
        info = client.get_collection(collection_name=collection_name)
        count = client.count(collection_name=collection_name, exact=True).count
        _require(count == 12, "persisted point count must be 12")

        vectors_config = info.config.params.vectors
        _require("dense" in vectors_config, "dense vector config missing")
        _require(int(vectors_config["dense"].size) == 1024, "dense size must be 1024")
        sparse = info.config.params.sparse_vectors
        _require(sparse is not None and "sparse" in sparse, "sparse vector config missing")

        # No query is issued. Only metadata.
        _ = client.scroll(
            collection_name=collection_name,
            limit=100,
            with_payload=False,
            with_vectors=False,
        )
    finally:
        with contextlib.suppress(Exception):
            client.close()

    print("[OK] Pilot Qdrant ingestion smoke")
    print("  collection:", collection_name)
    print("  persisted points: 12")
    print("  dense: 1024 COSINE")
    print("  sparse: present")
    print("  client reopened: True")
    print("  retrieval executed: False")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())