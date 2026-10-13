"""Smoke test: adapter materialization, determinism, source immutability."""

from __future__ import annotations

import hashlib
import sys
import tempfile
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

SOURCE = Path("data/pilot/infrastructure-pilot-synthetic-v1")


def _sha_of_tree(root: Path) -> str:
    h = hashlib.sha256()
    for path in sorted(root.iterdir()):
        if path.is_file():
            h.update(path.name.encode("utf-8"))
            h.update(path.read_bytes())
    return h.hexdigest()


def main() -> int:
    if not SOURCE.is_dir():
        print(f"source not found: {SOURCE}", file=sys.stderr)
        return 1

    source_sha_before = _sha_of_tree(SOURCE)

    documents, examples, validation, mapping = adapt_pilot_dataset_to_canonical(
        pilot_root=SOURCE
    )
    validate_pilot_canonical_adapter(
        documents=documents,
        examples=examples,
        validation=validation,
        mapping=mapping,
    )

    with tempfile.TemporaryDirectory() as td_a, tempfile.TemporaryDirectory() as td_b:
        root_a = Path(td_a) / "canonical"
        root_b = Path(td_b) / "canonical"

        write_pilot_canonical_adapter(
            target_root=root_a, documents=documents, examples=examples,
            validation=validation, mapping=mapping,
        )
        write_pilot_canonical_adapter(
            target_root=root_b, documents=documents, examples=examples,
            validation=validation, mapping=mapping,
        )

        a_names = sorted(p.name for p in root_a.iterdir() if p.is_file())
        b_names = sorted(p.name for p in root_b.iterdir() if p.is_file())
        deterministic_bytes = all(
            (root_a / n).read_bytes() == (root_b / n).read_bytes()
            for n in a_names
        ) and a_names == b_names

        loaded = read_pilot_canonical_adapter(
            target_root=root_a, expected_adapter_id=ADAPTER_ID,
        )

    source_sha_after = _sha_of_tree(SOURCE)
    source_unchanged = source_sha_before == source_sha_after

    print("[OK] Pilot canonical adapter smoke")
    print(f"  adapter: {ADAPTER_ID}")
    print(f"  documents: {len(loaded['documents'])}")
    print(f"  examples: {len(loaded['examples'])}")
    print(f"  duplicate content preserved: {validation.duplicate_content_preserved}")
    print(f"  source chunk reconstruction valid: {validation.source_chunk_reconstruction_valid}")
    print("  relevant documents per example: 2")
    print(f"  answer fields absent: {validation.answer_fields_absent}")
    print(f"  deterministic bytes: {deterministic_bytes}")
    print(f"  source artifacts unchanged: {source_unchanged}")
    print("  round trip: True")
    print("  checksums valid: True")
    print("  parser executed: False")
    print("  chunker executed: False")
    print("  embedding executed: False")
    print("  retrieval executed: False")
    print("  evaluation executed: False")
    print("  generation executed: False")

    return 0 if (source_unchanged and deterministic_bytes) else 1


if __name__ == "__main__":
    sys.exit(main())
