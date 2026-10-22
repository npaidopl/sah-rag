"""Fast smoke check for the Step 87 retrieval outputs."""

from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
EXECUTION_CONTRACT = REPO_ROOT / "baseline_audit" / "pilot_retrieval_execution_contract.yaml"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit("[FAIL] " + message)


def main() -> int:
    contract = yaml.safe_load(EXECUTION_CONTRACT.read_text(encoding="utf-8"))
    root = REPO_ROOT / contract["output"]["root"]
    _require(root.is_dir(), "retrieval output root missing")

    for name in contract["output"]["required_files"]:
        _require((root / name).exists(), "missing: " + name)

    from sah_rag.retrieval.pilot_persistence import (
        read_pilot_retrieval_artifacts,
    )

    loaded = read_pilot_retrieval_artifacts(
        root=root, expected_query_count=6
    )
    _require(loaded["checksums_valid"], "checksums not valid")

    print("[OK] Pilot hybrid retrieval smoke")
    print("  retrieval root:", str(root.relative_to(REPO_ROOT)))
    print("  query embeddings:", len(loaded["query_embeddings"]))
    print("  ranked children:", len(loaded["ranked_children"]))
    print("  ranked parents:", len(loaded["ranked_parents"]))
    print("  ranked documents:", len(loaded["ranked_documents"]))
    print("  checksums valid: True")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())