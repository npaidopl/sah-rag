"""Materialize the synthetic pilot dataset (Step 78)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from datasets.pilot_synthetic import (
    DATASET_ID,
    DATASET_REVISION,
    build_infrastructure_pilot_dataset,
)
from datasets.pilot_persistence import (
    read_infrastructure_pilot_dataset,
    write_infrastructure_pilot_dataset,
)

DEFAULT_ROOT = Path("data/pilot/infrastructure-pilot-synthetic-v1")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    root = Path(args.root)
    documents, chunks, queries, judgments = build_infrastructure_pilot_dataset()

    manifest = write_infrastructure_pilot_dataset(
        root=root,
        documents=documents,
        chunks=chunks,
        queries=queries,
        judgments=judgments,
        overwrite=args.overwrite,
    )
    loaded = read_infrastructure_pilot_dataset(
        root=root,
        expected_dataset_id=DATASET_ID,
        expected_dataset_revision=DATASET_REVISION,
    )

    print("[OK] Infrastructure pilot dataset materialized")
    print(f"  dataset id: {manifest.dataset_id}")
    print(f"  revision:   {manifest.dataset_revision}")
    print(f"  split:      {manifest.split}")
    print(f"  documents:  {manifest.document_count}")
    print(f"  unique content groups: {manifest.unique_content_count}")
    print(f"  duplicate-content pairs: {manifest.duplicate_content_pair_count}")
    print(f"  chunks:     {manifest.chunk_count}")
    print(f"  queries:    {manifest.query_count}")
    print(f"  judgments:  {manifest.judgment_count}")
    print(f"  relevant documents per query: {manifest.judgment_count // manifest.query_count}")
    print(f"  round trip: True ({len(loaded['documents'])} documents reloaded)")
    print("  checksums valid: True")
    print("  baseline execution performed: False")
    print("  model execution performed: False")
    print("  retrieval execution performed: False")
    print("  evaluation execution performed: False")
    print("  generation performed: False")
    return 0


if __name__ == "__main__":
    sys.exit(main())
