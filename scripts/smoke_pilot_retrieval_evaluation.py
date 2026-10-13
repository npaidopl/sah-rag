"""Fast smoke check for the Step 89 evaluation outputs."""

from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
EXECUTION_CONTRACT = (
    REPO_ROOT / "baseline_audit" / "pilot_evaluation_execution_contract.yaml"
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit("[FAIL] " + message)


def main() -> int:
    contract = yaml.safe_load(EXECUTION_CONTRACT.read_text(encoding="utf-8"))
    root = REPO_ROOT / contract["output"]["root"]
    _require(root.is_dir(), "evaluation output root missing")

    for name in contract["output"]["required_files"]:
        _require((root / name).exists(), "missing: " + name)

    from rag_system.evaluation.pilot_persistence import (
        read_pilot_evaluation_artifacts,
    )

    loaded = read_pilot_evaluation_artifacts(
        root=root, expected_query_count=6
    )
    _require(loaded["checksums_valid"], "checksums not valid")

    print("[OK] Pilot retrieval evaluation smoke")
    print("  retrieval cases:", len(loaded["retrieval_cases"]))
    print("  query metric rows:", len(loaded["query_metrics"]))
    print("  aggregate metric rows:", len(loaded["aggregate_metrics"]))
    print("  checksums valid: True")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())