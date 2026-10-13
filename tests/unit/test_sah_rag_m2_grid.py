"""Unit tests for SAH-RAG M2 chunking grid."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "run_sah_rag_m2_chunking_grid.py"


def _load():
    spec = importlib.util.spec_from_file_location("m2", SCRIPT)
    assert spec is not None and spec.loader is not None
    m = importlib.util.module_from_spec(spec)
    sys.modules["m2"] = m
    spec.loader.exec_module(m)
    return m


def test_grid_size():
    m = _load()
    assert len(m.GRID) == 9
    sizes = {cs for cs, _ in m.GRID}
    overlaps = {ov for _, ov in m.GRID}
    assert sizes == {256, 512, 1024}
    assert overlaps == {0, 32, 128}


def test_grid_includes_default():
    m = _load()
    assert (256, 32) in m.GRID


def test_grid_overlap_lt_size():
    m = _load()
    for cs, ov in m.GRID:
        assert ov < cs


def test_variant_dir_naming():
    m = _load()
    p = m.variant_dir("clerk", 512, 32)
    assert p.name == "c512_o32"
    assert p.parent.name == "grid-chunks-v1"


def test_datasets_match_contract():
    m = _load()
    assert m.DATASETS == ("clerk", "cuad", "greek_bar_bench", "lara")