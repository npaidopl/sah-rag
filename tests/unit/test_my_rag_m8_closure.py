"""Unit tests for My_RAG M8 closure."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "run_my_rag_m8_closure.py"


def _load():
    spec = importlib.util.spec_from_file_location("m8", SCRIPT)
    assert spec is not None and spec.loader is not None
    m = importlib.util.module_from_spec(spec)
    sys.modules["m8"] = m
    spec.loader.exec_module(m)
    return m


def test_frozen_config_readable():
    m = _load()
    frozen = m.load_yaml(
        Path("baseline_audit") / "my_rag_m6_frozen_config.yaml"
    )
    assert frozen["selected_config_id"] == "c256_o32_hybrid_rrf_k200"


def test_verdicts_cover_all_hypotheses():
    m = _load()
    curves = m.load_json(
        Path("outputs/external_my_rag/_m4b_response_curves.json")
    )
    reduction = m.load_json(
        Path("outputs/external_my_rag/_m4b_reduction.json")
    )
    verdicts = m.hypothesis_verdicts(curves, reduction)
    for h in ("H1", "H2", "H3", "H4", "H5", "H6", "H7"):
        assert h in verdicts
        assert "verdict" in verdicts[h]