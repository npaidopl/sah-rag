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

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_ENV_FILE = _PROJECT_ROOT / ".env"


def _ensure_env_loaded() -> None:
    """Load .env if RAG_CANONICAL_ROOT is not already set.

    Why this exists: `sah_rag.canonical.paths.get_canonical_root()` calls
    an internal `load_project_environment()` whose behavior depends on the
    current working directory (and sometimes on package import order).
    When scripts run from arbitrary CWDs (e.g., GPU server, scripts/),
    that fails silently. This helper loads .env explicitly from the project
    root so the module works regardless of CWD.
    """
    if os.getenv("RAG_CANONICAL_ROOT"):
        return
    if not _ENV_FILE.exists():
        return
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(_ENV_FILE, override=False)


_ensure_env_loaded()


from sah_rag.canonical import (  # noqa: E402
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


def _resolve_root(root: Path | None = None) -> Path:
    """Resolve RAG_CANONICAL_ROOT with fallbacks."""
    _ensure_env_loaded()
    if root is not None:
        return Path(root)
    try:
        from sah_rag.canonical.paths import get_canonical_root
        return get_canonical_root()
    except RuntimeError:
        # Last-resort fallback: project_root / processed / canonical
        return _PROJECT_ROOT / "data" / "processed" / "canonical"


def load_dataset(
    dataset_id: str,
    root: Path | None = None,
) -> UnifiedDataset:
    """Load a canonical dataset from `processed/canonical/<dataset_id>/`.

    Required files under `<root>/<dataset_id>/`:
      - documents.jsonl
      - examples.jsonl

    Raises:
        FileNotFoundError: if the dataset directory does not exist.
        ValueError: if required jsonl files are missing.
    """
    _ensure_env_loaded()
    paths = resolve_dataset_paths(dataset_id, root=_resolve_root(root))
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
    resolved = _resolve_root(root)
    if not resolved.exists():
        return []
    return sorted(
        p.name
        for p in resolved.iterdir()
        if p.is_dir() and (p / "examples.jsonl").exists()
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
