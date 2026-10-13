"""Validate the pilot parser+chunker manifest."""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

MANIFEST = Path("baseline_audit/pilot_parser_chunker_manifest.yaml")


def main() -> int:
    data = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    ex = data["execution"]
    if ex["execution_id"] != "pilot-parser-chunker-v1":
        raise SystemExit("execution_id changed")
    if ex["status"] not in ("PLANNED", "SUCCEEDED"):
        raise SystemExit("status invalid")

    ap = data["approval"]
    if ap["parser_execution_approved"] is not True:
        raise SystemExit("parser_execution_approved must be true")
    if ap["chunker_execution_approved"] is not True:
        raise SystemExit("chunker_execution_approved must be true")
    for k in (
        "embedding_execution_approved", "index_execution_approved",
        "retrieval_execution_approved", "evaluation_execution_approved",
        "generation_execution_approved",
    ):
        if ap[k] is not False:
            raise SystemExit(f"approval.{k} must be false")

    print("[OK] Pilot parser+chunker manifest")
    print(f"  execution id: {ex['execution_id']}")
    print(f"  status: {ex['status']}")
    print(f"  parser approved: {ap['parser_execution_approved']}")
    print(f"  chunker approved: {ap['chunker_execution_approved']}")
    print("  downstream approved: false")
    return 0


if __name__ == "__main__":
    sys.exit(main())
