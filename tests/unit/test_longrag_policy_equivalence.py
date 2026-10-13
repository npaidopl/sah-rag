"""Policy equivalence invariants for the source-corrected LongRAG pilot."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).parents[2]
PILOT_ROOT = ROOT / "outputs" / "longrag_clerk_source_corrected_pilot"
SD_PATH = PILOT_ROOT / "LONGRAG-SD-4096" / "units.jsonl"
PC_PATH = PILOT_ROOT / "LONGRAG-PC-4096" / "units.jsonl"


def _load(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        pytest.skip(f"Pilot output missing: {path}")
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _content_key(unit: dict[str, Any]) -> tuple[Any, ...]:
    """Identity that ignores ordering and policy-specific identifiers."""
    return (
        tuple(unit["member_doc_ids"]),
        tuple(unit["member_parent_ids"]),
        tuple(unit["member_chunk_ids"]),
        unit["text"],
        unit["token_count"],
    )


def test_single_document_and_parent_group_are_content_equivalent() -> None:
    """For canonical-document input, both policies must yield same content."""
    sd = _load(SD_PATH)
    pc = _load(PC_PATH)

    assert len(sd) == len(pc)

    sd_keys = sorted(_content_key(u) for u in sd)
    pc_keys = sorted(_content_key(u) for u in pc)

    assert sd_keys == pc_keys


def test_long_unit_ids_encode_construction_policy() -> None:
    """Policy is part of identity by design, so ID spaces must be disjoint."""
    sd = _load(SD_PATH)
    pc = _load(PC_PATH)

    sd_ids = {u["long_unit_id"] for u in sd}
    pc_ids = {u["long_unit_id"] for u in pc}

    assert sd_ids.isdisjoint(pc_ids)
    assert all(
        "single-document-if-within-budget" in i for i in sd_ids
    )
    assert all(
        "structure-preserving-parent-groups" in i for i in pc_ids
    )