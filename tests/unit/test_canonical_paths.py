"""Tests for canonical path resolution."""

from __future__ import annotations

from pathlib import Path

from sah_rag.canonical.paths import (
    list_dataset_ids,
    resolve_dataset_paths,
)


def create_dataset(root: Path, dataset_id: str) -> None:
    """Create a minimal canonical dataset directory."""

    dataset_root = root / dataset_id
    dataset_root.mkdir(parents=True)

    for file_name in (
        "documents.jsonl",
        "examples.jsonl",
        "validation.json",
        "manifest.json",
    ):
        (dataset_root / file_name).write_text(
            "",
            encoding="utf-8",
        )


def test_list_dataset_ids_is_sorted(tmp_path: Path) -> None:
    """Dataset IDs should be returned in stable order."""

    create_dataset(tmp_path, "ruler")
    create_dataset(tmp_path, "clerk")

    assert list_dataset_ids(tmp_path) == (
        "clerk",
        "ruler",
    )


def test_resolve_complete_dataset(tmp_path: Path) -> None:
    """All canonical paths should resolve for a complete dataset."""

    create_dataset(tmp_path, "cuad")

    paths = resolve_dataset_paths(
        "cuad",
        root=tmp_path,
    )

    assert paths.dataset_id == "cuad"
    assert paths.documents.name == "documents.jsonl"
    assert paths.examples.name == "examples.jsonl"
    assert paths.validation.name == "validation.json"
    assert paths.manifest.name == "manifest.json"
    assert paths.is_complete
    assert paths.missing_files() == ()