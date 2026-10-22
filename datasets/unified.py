"""Unified dataset loader for SAH-RAG v2.

This is a THIN WRAPPER around `sah_rag.canonical`. It exists so that
baselines have a single import path (`from datasets.unified import ...`)
without knowing the internal structure of `sah_rag.canonical`.

Do NOT duplicate loading logic here. Delegate to sah_rag.canonical.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from sah_rag.canonical import (
    CanonicalDatasetPaths,
    CanonicalDocument,
    CanonicalExample,
    iter_canonical_documents,
    iter_canonical_examples,
    resolve_dataset_paths,
)


@dataclass(frozen=True)
class UnifiedDataset:
    """A dataset ready to feed a baseline."""

    dataset_id: str
    documents: list[CanonicalDocument]
    examples: list[CanonicalExample]
    metadata: dict[str, Any] = field(default_factory=dict)

    def iter_examples(self) -> Iterator[CanonicalExample]:
        yield from self.examples


def load_dataset(
    dataset_id: str,
    root: Path | None = None,
) -> UnifiedDataset:
    """Load a canonical dataset from `processed/canonical/<dataset_id>/`.

    Uses `RAG_CANONICAL_ROOT` env var if `root` is None.

    Required files under `<root>/<dataset_id>/`:
      - documents.jsonl
      - examples.jsonl
      - manifest.json  (optional, for metadata)

    Raises:
        FileNotFoundError: if the dataset directory does not exist.
        ValueError: if required jsonl files are missing.
    """
    paths = resolve_dataset_paths(dataset_id, root=root)
    documents = list(iter_canonical_documents(paths))
    examples = list(iter_canonical_examples(paths))
    return UnifiedDataset(
        dataset_id=dataset_id,
        documents=documents,
        examples=examples,
        metadata={
            "paths": str(paths.root),
            "num_documents": len(documents),
            "num_examples": len(examples),
        },
    )


def list_available_datasets(root: Path | None = None) -> list[str]:
    """List dataset IDs present under `processed/canonical/`."""
    from sah_rag.canonical.paths import get_canonical_root
    if root is None:
        root = get_canonical_root()
    if not root.exists():
        return []
    return sorted(
        p.name for p in root.iterdir() if p.is_dir() and (p / "examples.jsonl").exists()
    )

__all__ = [
    "CanonicalDatasetPaths",
    "CanonicalDocument",
    "CanonicalExample",
    "UnifiedDataset",
    "list_available_datasets",
    "load_dataset",
    "resolve_dataset_paths",
]
