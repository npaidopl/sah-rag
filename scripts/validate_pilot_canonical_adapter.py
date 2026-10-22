"""Validate the materialized pilot canonical adapter target."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sah_rag.canonical.pilot_adapter import (
    ADAPTER_ID,
    validate_pilot_canonical_adapter,
)
from sah_rag.canonical.pilot_adapter_persistence import (
    read_pilot_canonical_adapter,
)

DEFAULT_TARGET = Path("processed/canonical/infrastructure-pilot-synthetic-v1")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default=str(DEFAULT_TARGET))
    args = parser.parse_args()

    loaded = read_pilot_canonical_adapter(
        target_root=Path(args.target),
        expected_adapter_id=ADAPTER_ID,
    )

    validate_pilot_canonical_adapter(
        documents=loaded["documents"],
        examples=loaded["examples"],
        validation=loaded["validation"],
        mapping=loaded["mapping"],
    )

    print("[OK] Pilot canonical adapter")
    print(f"  target: {Path(args.target).resolve()}")
    print(f"  adapter: {loaded['manifest'].adapter_id}")
    print(f"  version: {loaded['manifest'].adapter_version}")
    print(f"  documents: {len(loaded['documents'])}")
    print(f"  examples: {len(loaded['examples'])}")
    print(f"  validation.status: {loaded['validation'].status}")
    v = loaded['validation']
    print(f"  duplicate_content_preserved: {v.duplicate_content_preserved}")
    print(f"  source_chunk_reconstruction_valid: {v.source_chunk_reconstruction_valid}")
    print(f"  relevant_document_mapping_valid: {v.relevant_document_mapping_valid}")
    print(f"  answer_fields_absent: {loaded['validation'].answer_fields_absent}")
    print(f"  source_immutable: {loaded['validation'].source_immutable}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
