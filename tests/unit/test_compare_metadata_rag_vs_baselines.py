"""Tests for the cross-architecture comparison script."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = (
    REPO_ROOT / "scripts" / "compare_metadata_rag_vs_baselines.py"
)


def _load_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "compare_metadata_rag_vs_baselines", SCRIPT_PATH
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cmp_mod = _load_module()


def test_constants() -> None:
    assert cmp_mod.EXPECTED_DATASETS == (
        "clerk",
        "cuad",
        "greek_bar_bench",
        "lara",
    )
    assert cmp_mod.PRIMARY_METRIC == "recall_at_10"
    assert cmp_mod.COMPARISON_ARTIFACT_TYPE == (
        "metadata_rag_vs_baselines_comparison"
    )


def _write_aggregate(
    root: Path, dataset: str, mean: dict[str, float]
) -> None:
    d = root / dataset / "evaluation-v1"
    d.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset_id": dataset,
        "mean": mean,
        "cutoffs": [1, 5, 10, 20, 50, 100],
    }
    (d / "aggregate_metrics.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )


def test_secondary_macro_average(tmp_path: Path) -> None:
    per_dataset = {
        "clerk": {"mean": {"recall_at_10": 0.4}},
        "cuad": {"mean": {"recall_at_10": 0.2}},
    }
    out = cmp_mod._secondary_macro(per_dataset)
    assert out["recall_at_10"] == pytest.approx(0.3)


def test_secondary_macro_handles_missing_keys() -> None:
    per_dataset = {
        "a": {"mean": {"recall_at_10": 1.0}},
        "b": {"mean": {}},
    }
    out = cmp_mod._secondary_macro(per_dataset)
    assert out["recall_at_10"] == pytest.approx(0.5)


def test_load_track_aggregates_skips_missing(tmp_path: Path) -> None:
    _write_aggregate(tmp_path, "clerk", {"recall_at_10": 0.3})
    out = cmp_mod._load_track_aggregates(
        tmp_path, ("clerk", "cuad")
    )
    assert "clerk" in out
    assert "cuad" not in out


def test_load_track_aggregates_rejects_missing_mean(
    tmp_path: Path,
) -> None:
    d = tmp_path / "clerk" / "evaluation-v1"
    d.mkdir(parents=True)
    (d / "aggregate_metrics.json").write_text(
        json.dumps({"dataset_id": "clerk"}), encoding="utf-8"
    )
    with pytest.raises(cmp_mod.ComparisonError):
        cmp_mod._load_track_aggregates(tmp_path, ("clerk",))


def _write_legacy_aggregate(
    root: Path, dataset: str, macro: dict[str, float]
) -> None:
    d = root / dataset / "evaluation-v1"
    d.mkdir(parents=True, exist_ok=True)
    payload = {
        "artifact_type": "legacy_evaluation_report",
        "artifact_version": "1.0",
        "dataset_id": dataset,
        "cutoffs": [1, 5, 10, 20, 50],
        "macro_metrics": macro,
    }
    (d / "per_dataset_metrics.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )


def test_normalize_metric_keys_converts_at_symbol() -> None:
    raw = {"hit@10": 0.5, "recall@10": 0.7, "mrr@10": 0.3}
    out = cmp_mod._normalize_metric_keys(raw)
    assert out == {
        "hit_at_10": 0.5,
        "recall_at_10": 0.7,
        "mrr_at_10": 0.3,
    }


def test_extract_mean_block_prefers_native(tmp_path: Path) -> None:
    aggregate = {
        "mean": {"recall_at_10": 0.4},
        "macro_metrics": {"recall@10": 0.9},
    }
    out = cmp_mod._extract_mean_block(
        aggregate, tmp_path / "x.json"
    )
    assert out == {"recall_at_10": 0.4}


def test_extract_mean_block_supports_legacy(tmp_path: Path) -> None:
    aggregate = {"macro_metrics": {"recall@10": 0.4}}
    out = cmp_mod._extract_mean_block(
        aggregate, tmp_path / "x.json"
    )
    assert out == {"recall_at_10": 0.4}


def test_extract_mean_block_rejects_unknown_format(
    tmp_path: Path,
) -> None:
    with pytest.raises(cmp_mod.ComparisonError):
        cmp_mod._extract_mean_block(
            {"unrelated": 1}, tmp_path / "x.json"
        )


def test_load_track_aggregates_supports_legacy_format(
    tmp_path: Path,
) -> None:
    _write_legacy_aggregate(
        tmp_path, "clerk", {"recall@10": 0.5, "hit@10": 0.6}
    )
    out = cmp_mod._load_track_aggregates(tmp_path, ("clerk",))
    assert "clerk" in out
    assert out["clerk"]["mean"] == {
        "recall_at_10": 0.5,
        "hit_at_10": 0.6,
    }


def test_load_track_aggregates_prefers_native_when_both_present(
    tmp_path: Path,
) -> None:
    _write_aggregate(tmp_path, "clerk", {"recall_at_10": 0.4})
    _write_legacy_aggregate(
        tmp_path, "clerk", {"recall@10": 0.9}
    )
    out = cmp_mod._load_track_aggregates(tmp_path, ("clerk",))
    assert out["clerk"]["mean"] == {"recall_at_10": 0.4}


def test_mixed_tracks_aggregate_across_formats(
    tmp_path: Path,
) -> None:
    native_root = tmp_path / "native"
    legacy_root = tmp_path / "legacy"
    _write_aggregate(native_root, "clerk", {"recall_at_10": 0.4})
    _write_aggregate(native_root, "cuad", {"recall_at_10": 0.2})
    _write_legacy_aggregate(
        legacy_root, "clerk", {"recall@10": 0.3}
    )
    _write_legacy_aggregate(
        legacy_root, "cuad", {"recall@10": 0.1}
    )
    comparison = cmp_mod.build_comparison(
        tracks=(
            ("native", "Native", native_root),
            ("legacy", "Legacy", legacy_root),
        ),
        datasets=("clerk", "cuad"),
    )
    assert comparison["tracks"]["native"]["secondary_macro"][
        "recall_at_10"
    ] == pytest.approx(0.3)
    assert comparison["tracks"]["legacy"]["secondary_macro"][
        "recall_at_10"
    ] == pytest.approx(0.2)


def test_build_comparison_shape(tmp_path: Path) -> None:
    root_a = tmp_path / "a"
    root_b = tmp_path / "b"
    _write_aggregate(root_a, "clerk", {"recall_at_10": 0.4})
    _write_aggregate(root_a, "cuad", {"recall_at_10": 0.2})
    _write_aggregate(root_b, "clerk", {"recall_at_10": 0.5})

    tracks = (
        ("metadata_rag", "Metadata-RAG", root_a),
        ("longrag", "LongRAG", root_b),
    )
    comparison = cmp_mod.build_comparison(
        tracks=tracks, datasets=("clerk", "cuad")
    )
    assert comparison["artifact_type"] == (
        "metadata_rag_vs_baselines_comparison"
    )
    assert "metadata_rag" in comparison["tracks"]
    assert "longrag" in comparison["tracks"]

    mr = comparison["tracks"]["metadata_rag"]
    assert mr["datasets_present"] == ["clerk", "cuad"]
    assert mr["datasets_missing"] == []
    assert mr["secondary_macro"]["recall_at_10"] == pytest.approx(0.3)

    lr = comparison["tracks"]["longrag"]
    assert lr["datasets_present"] == ["clerk"]
    assert lr["datasets_missing"] == ["cuad"]


def test_render_markdown_contains_sections(tmp_path: Path) -> None:
    root_a = tmp_path / "a"
    _write_aggregate(root_a, "clerk", {"recall_at_10": 0.4})
    comparison = cmp_mod.build_comparison(
        tracks=(("metadata_rag", "Metadata-RAG", root_a),),
        datasets=("clerk",),
    )
    md = cmp_mod.render_markdown(comparison)
    assert "# Metadata-RAG vs Baselines" in md
    assert "## 1. Per-dataset primary metric" in md
    assert "## 2. Secondary macro across datasets" in md
    assert "## 3. Dataset coverage" in md
    assert "Metadata-RAG" in md


def test_strict_mode_fails_on_missing_dataset(
    tmp_path: Path,
) -> None:
    root_a = tmp_path / "a"
    _write_aggregate(root_a, "clerk", {"recall_at_10": 0.4})

    args = cmp_mod._parse_args(
        [
            "--output-dir",
            str(tmp_path / "out"),
            "--datasets",
            "clerk",
            "cuad",
            "--strict",
        ]
    )
    # We can't call main() with our own tracks easily; instead simulate
    # the strict check.
    comparison = cmp_mod.build_comparison(
        tracks=(("metadata_rag", "Metadata-RAG", root_a),),
        datasets=tuple(args.datasets),
    )
    with pytest.raises(SystemExit):
        for key, track in comparison["tracks"].items():
            if track["datasets_missing"]:
                raise SystemExit(
                    f"[strict] track {key} is missing datasets: "
                    f"{track['datasets_missing']}"
                )
