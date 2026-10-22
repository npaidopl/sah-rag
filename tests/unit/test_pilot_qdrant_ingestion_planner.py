"""Tests for the pilot Qdrant ingestion planner (deterministic, no Qdrant)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PLANNER = REPO_ROOT / "scripts" / "plan_pilot_qdrant_ingestion.py"


def _load_planner():
    spec = importlib.util.spec_from_file_location("_pilot_qdrant_planner", PLANNER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_deterministic_point_id_for_same_chunk():
    from sah_rag.storage.qdrant import build_qdrant_point_id

    a = build_qdrant_point_id("chunk-X")
    b = build_qdrant_point_id("chunk-X")
    assert a == b


def test_twelve_unique_point_ids():
    from sah_rag.storage.qdrant import build_qdrant_point_id

    chunk_ids = [f"chunk-{i:05d}" for i in range(12)]
    point_ids = [build_qdrant_point_id(c) for c in chunk_ids]
    assert len(set(point_ids)) == 12


def test_configuration_hash_is_deterministic():
    planner = _load_planner()
    contract = {
        "source": {"dataset": "infrastructure-pilot-synthetic-v1"},
    }
    catalog = {"catalog_sha256": "a" * 64}
    h1 = planner.compute_informational_hash(contract, catalog)
    h2 = planner.compute_informational_hash(contract, catalog)
    assert h1 == h2
    assert len(h1) == 64
    int(h1, 16)


def test_physical_collection_name_is_deterministic():
    planner = _load_planner()
    name_a = planner.compute_physical_name(
        "infrastructure-pilot-synthetic-v1", "b" * 64
    )
    name_b = planner.compute_physical_name(
        "infrastructure-pilot-synthetic-v1", "b" * 64
    )
    assert name_a == name_b
    assert name_a.startswith("rag_infrastructure_pilot_synthetic_v1_")


def test_planner_source_does_not_instantiate_qdrant_client():
    source = PLANNER.read_text(encoding="utf-8")
    assert "QdrantClient(" not in source


def test_planner_source_does_not_create_storage_root():
    source = PLANNER.read_text(encoding="utf-8")
    # must not call mkdir, touch, open(..., "w"), etc. on storage path
    assert ".mkdir(" not in source
    assert "storage_root.mkdir" not in source