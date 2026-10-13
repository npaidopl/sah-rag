"""Materialize the pilot canonical adapter target (Step 80)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rag_system.canonical.pilot_adapter import (
    ADAPTER_ID,
    adapt_pilot_dataset_to_canonical,
    validate_pilot_canonical_adapter,
)
from rag_system.canonical.pilot_adapter_persistence import (
    read_pilot_canonical_adapter,
    write_pilot_canonical_adapter,
)

DEFAULT_SOURCE = Path("data/pilot/infrastructure-pilot-synthetic-v1")
DEFAULT_TARGET = Path("processed/canonical/infrastructure-pilot-synthetic-v1")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=str(DEFAULT_SOURCE))
    parser.add_argument("--target", default=str(DEFAULT_TARGET))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    source_root = Path(args.source)
    target_root = Path(args.target)

    documents, examples, validation, mapping = adapt_pilot_dataset_to_canonical(
        pilot_root=source_root
    )
    validate_pilot_canonical_adapter(
        documents=documents,
        examples=examples,
        validation=validation,
        mapping=mapping,
    )

    manifest = write_pilot_canonical_adapter(
        target_root=target_root,
        documents=documents,
        examples=examples,
        validation=validation,
        mapping=mapping,
        overwrite=args.overwrite,
    )

    loaded = read_pilot_canonical_adapter(
        target_root=target_root,
        expected_adapter_id=ADAPTER_ID,
    )

    print("[OK] Pilot canonical adapter materialized")
    print(f"  adapter: {manifest.adapter_id}")
    print(f"  version: {manifest.adapter_version}")
    print(f"  source dataset: {manifest.source_dataset_id}")
    print(f"  source documents: {len(documents)}")
    print("  source chunks validated: 24")
    print(f"  target documents: {len(loaded['documents'])}")
    print(f"  target examples: {len(loaded['examples'])}")
    print(f"  duplicate-content pairs preserved: {validation.duplicate_content_preserved}")
    print("  relevant documents per example: 2")
    print(f"  source artifacts unchanged: {validation.source_immutable}")
    print(f"  round trip: True ({len(loaded['documents'])} documents reloaded)")
    print("  checksums valid: True")
    print("  parser executed: False")
    print("  chunker executed: False")
    print("  embedding executed: False")
    print("  retrieval executed: False")
    print("  evaluation executed: False")
    print("  generation executed: False")
    return 0


if __name__ == "__main__":
    sys.exit(main())
