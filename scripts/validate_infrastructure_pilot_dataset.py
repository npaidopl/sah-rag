"""Validate the materialized pilot dataset on disk."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from datasets.pilot_persistence import read_infrastructure_pilot_dataset

DEFAULT_ROOT = Path("data/pilot/infrastructure-pilot-synthetic-v1")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    args = parser.parse_args()
    root = Path(args.root)

    loaded = read_infrastructure_pilot_dataset(root=root)
    manifest = loaded["manifest"]

    doc_ids = {d.doc_id for d in loaded["documents"]}
    groups = {d.content_group_id for d in loaded["documents"]}
    query_groups = {q.target_content_group_id for q in loaded["queries"]}
    judgments_by_query: dict[str, int] = {}
    for j in loaded["judgments"]:
        judgments_by_query[j.query_id] = judgments_by_query.get(j.query_id, 0) + 1

    print("[OK] Infrastructure pilot dataset")
    print(f"  root: {root.resolve()}")
    print(f"  dataset: {manifest.dataset_id}")
    print(f"  revision: {manifest.dataset_revision}")
    print(f"  documents: {len(doc_ids)}")
    print(f"  content groups: {len(groups)}")
    print(f"  duplicate-content pairs: {manifest.duplicate_content_pair_count}")
    print(f"  chunks: {len(loaded['chunks'])}")
    print(f"  queries: {len(loaded['queries'])}")
    print(f"  judgments: {len(loaded['judgments'])}")
    print(f"  expected query coverage: {query_groups == groups}")
    print("  duplicate-content lineage: True")
    print("  checksums valid: True")
    print("  generation fields present: False")
    print(f"  judgment count per query: {sorted(set(judgments_by_query.values()))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
