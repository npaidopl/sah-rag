"""Validate the pilot Qdrant collection plan."""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

PATH = Path("baseline_audit/pilot_qdrant_collection_plan.yaml")


def main() -> int:
    data = yaml.safe_load(PATH.read_text(encoding="utf-8"))
    if data["backend"]["type"] != "Qdrant":
        raise SystemExit("backend.type must be Qdrant")
    if data["backend"]["mode"] != "local_persistence":
        raise SystemExit("backend.mode must be local_persistence")

    d = data["vectors"]["dense"]
    if d["dimension"] != 1024:
        raise SystemExit("dense dimension must be 1024")
    if d["distance"] != "COSINE":
        raise SystemExit("dense distance must be COSINE")
    if data["vectors"]["sparse"]["name"] != "sparse":
        raise SystemExit("sparse vector name must be 'sparse'")

    safety = data["safety"]
    if safety["collection_creation_approved"] is not False:
        raise SystemExit("collection_creation_approved must be false")
    if safety["vector_ingestion_approved"] is not False:
        raise SystemExit("vector_ingestion_approved must be false")

    print("[OK] Pilot Qdrant collection plan")
    print(f"  backend: {data['backend']['type']} {data['backend']['mode']}")
    print(f"  dense vector: {d['name']} {d['dimension']} {d['distance']}")
    print(f"  sparse vector: {data['vectors']['sparse']['name']}")
    print(f"  collection creation: {safety['collection_creation_approved']}")
    print(f"  vector ingestion: {safety['vector_ingestion_approved']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
