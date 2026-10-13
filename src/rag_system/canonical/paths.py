"""Canonical dataset path resolution."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

REQUIRED_DATASET_FILES = (
    "documents.jsonl",
    "examples.jsonl",
    "validation.json",
    "manifest.json",
)


@dataclass(frozen=True, slots=True)
class CanonicalDatasetPaths:
    """Resolved paths for one canonical dataset."""

    dataset_id: str
    root: Path
    documents: Path
    examples: Path
    validation: Path
    manifest: Path

    def missing_files(self) -> tuple[Path, ...]:
        """Return required files that do not exist."""

        paths = (
            self.documents,
            self.examples,
            self.validation,
            self.manifest,
        )
        return tuple(path for path in paths if not path.is_file())

    @property
    def is_complete(self) -> bool:
        """Return whether all required files exist."""

        return not self.missing_files()


def get_project_root() -> Path:
    """Return the repository root inferred from this source file."""

    return Path(__file__).resolve().parents[3]


def load_project_environment() -> None:
    """Load the repository-local .env file when available."""

    env_path = get_project_root() / ".env"
    load_dotenv(dotenv_path=env_path, override=False)


def get_canonical_root() -> Path:
    """Resolve and validate RAG_CANONICAL_ROOT."""

    load_project_environment()

    raw_path = os.getenv("RAG_CANONICAL_ROOT")

    if not raw_path:
        raise RuntimeError(
            "RAG_CANONICAL_ROOT is not configured. "
            "Create a local .env file from .env.example."
        )

    root = Path(raw_path).expanduser().resolve()

    if not root.exists():
        raise FileNotFoundError(
            f"Canonical root does not exist: {root}"
        )

    if not root.is_dir():
        raise NotADirectoryError(
            f"Canonical root is not a directory: {root}"
        )

    return root


def list_dataset_ids(root: Path | None = None) -> tuple[str, ...]:
    """List canonical dataset directory names in stable order."""

    canonical_root = root or get_canonical_root()

    dataset_ids = sorted(
        path.name
        for path in canonical_root.iterdir()
        if path.is_dir()
    )

    return tuple(dataset_ids)


def resolve_dataset_paths(
    dataset_id: str,
    root: Path | None = None,
) -> CanonicalDatasetPaths:
    """Resolve all required paths for one dataset."""

    canonical_root = root or get_canonical_root()
    dataset_root = canonical_root / dataset_id

    if not dataset_root.is_dir():
        raise FileNotFoundError(
            f"Canonical dataset directory does not exist: {dataset_root}"
        )

    return CanonicalDatasetPaths(
        dataset_id=dataset_id,
        root=dataset_root,
        documents=dataset_root / "documents.jsonl",
        examples=dataset_root / "examples.jsonl",
        validation=dataset_root / "validation.json",
        manifest=dataset_root / "manifest.json",
    )