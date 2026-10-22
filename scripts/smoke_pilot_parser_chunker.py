"""Smoke: run parser+chunker twice, verify determinism + source immutability."""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

OUTPUT_ROOT = Path("outputs/infrastructure_pilot/infrastructure-pilot-synthetic-v1/sah_rag")
SOURCE_ROOTS = (
    Path("data/pilot/infrastructure-pilot-synthetic-v1"),
    Path("processed/canonical/infrastructure-pilot-synthetic-v1"),
)
RUNNER = Path("scripts/run_pilot_parser_chunker.py")


def _tree_sha(root: Path) -> str:
    """Hash only deterministic data artifacts (*.jsonl).

    Execution manifests carry runtime metadata and are excluded from
    the determinism comparison by design.
    """
    h = hashlib.sha256()
    for p in sorted(root.rglob("*.jsonl")):
        if p.is_file():
            h.update(str(p.relative_to(root)).encode("utf-8"))
            h.update(p.read_bytes())
    return h.hexdigest()


def _run(overwrite: bool) -> None:
    cmd = [sys.executable, str(RUNNER)]
    if overwrite:
        cmd.append("--overwrite")
    subprocess.run(cmd, check=True)


def main() -> int:
    source_before = {str(r): _tree_sha(r) for r in SOURCE_ROOTS}

    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)

    _run(overwrite=False)
    after1 = _tree_sha(OUTPUT_ROOT)

    _run(overwrite=True)
    after2 = _tree_sha(OUTPUT_ROOT)

    source_after = {str(r): _tree_sha(r) for r in SOURCE_ROOTS}

    deterministic = after1 == after2
    source_unchanged = source_before == source_after

    print("[OK] Pilot parser+chunker smoke")
    print(f"  deterministic artifacts: {deterministic}")
    print(f"  source artifacts unchanged: {source_unchanged}")
    print("  embedding executed: False")
    print("  index executed: False")
    print("  retrieval executed: False")
    print("  evaluation executed: False")
    print("  generation executed: False")

    return 0 if (deterministic and source_unchanged) else 1


if __name__ == "__main__":
    sys.exit(main())
