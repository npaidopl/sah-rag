"""Validate the pilot embedding manifest."""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

PATH = Path("baseline_audit/pilot_embedding_manifest.yaml")
ALLOWED_CACHE = {"UNVERIFIED", "VERIFIED", "REVISION_MISSING", "NOT_AVAILABLE", "PARTIAL"}


def main() -> int:
    data = yaml.safe_load(PATH.read_text(encoding="utf-8"))
    ex = data["execution"]
    if ex["execution_id"] != "pilot-bge-m3-embedding-v1":
        raise SystemExit("execution_id changed")
    if ex["status"] not in ("PLANNED", "SUCCEEDED"):
        raise SystemExit("status invalid")

    ap = data["approval"]
    if ap["cache_inspection_approved"] is not True:
        raise SystemExit("cache_inspection_approved must be true")
    if ap["model_download_approved"] is not False:
        raise SystemExit("model_download_approved must be false")
    for k in (
        "embedding_execution_approved", "index_execution_approved",
        "retrieval_execution_approved", "evaluation_execution_approved",
        "generation_execution_approved",
    ):
        if ap[k] is not False:
            raise SystemExit(f"approval.{k} must be false")

    cs = data["model"]["cache_status"]
    if cs not in ALLOWED_CACHE:
        raise SystemExit("cache_status invalid")

    print("[OK] Pilot embedding manifest")
    print(f"  execution id: {ex['execution_id']}")
    print(f"  status: {ex['status']}")
    print(f"  cache status: {cs}")
    print(f"  embedding execution approved: {ap['embedding_execution_approved']}")
    print(f"  model download approved: {ap['model_download_approved']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
