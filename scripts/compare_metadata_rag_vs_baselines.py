# ruff: noqa: E402
"""Cross-architecture comparison over the 4-dataset scope.

Reads the per-dataset aggregate metrics of three tracks:

  - metadata_rag   outputs/external_metadata_rag/<ds>/evaluation-v1/
  - longrag        outputs/external_longrag/<ds>/evaluation-v1/
  - my_rag         outputs/external_my_rag/<ds>/evaluation-v1/

Computes the macro-then-secondary-macro aggregation over the four
datasets for every cutoff and every metric, and writes a single
comparison artifact (JSON + Markdown) under baseline_audit/reports/.

The script reads ONLY aggregate metrics; it never touches per-query
files, ground truth, or raw retrieval. It performs no computation
that could depend on hidden data.

Missing tracks or datasets are recorded as ``MISSING`` and do not
abort the comparison; a partial report is still useful for progress
inspection. Full reports require every track to be present.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Final

REPO_ROOT = Path(__file__).resolve().parents[1]

EXPECTED_DATASETS: Final[tuple[str, ...]] = (
    "clerk",
    "cuad",
    "greek_bar_bench",
    "lara",
)

CUTOFFS: Final[tuple[int, ...]] = (1, 5, 10, 20, 50, 100)

# (artifact key, human label)
METRIC_KEYS: Final[tuple[tuple[str, str], ...]] = (
    ("hit_at_1", "Hit@1"),
    ("hit_at_5", "Hit@5"),
    ("hit_at_10", "Hit@10"),
    ("hit_at_20", "Hit@20"),
    ("hit_at_50", "Hit@50"),
    ("hit_at_100", "Hit@100"),
    ("precision_at_10", "Precision@10"),
    ("recall_at_1", "Recall@1"),
    ("recall_at_5", "Recall@5"),
    ("recall_at_10", "Recall@10"),
    ("recall_at_20", "Recall@20"),
    ("recall_at_50", "Recall@50"),
    ("recall_at_100", "Recall@100"),
    ("mrr_at_10", "MRR@10"),
    ("mrr_at_50", "MRR@50"),
    ("ndcg_at_10", "nDCG@10"),
    ("ndcg_at_50", "nDCG@50"),
)

PRIMARY_METRIC: Final[str] = "recall_at_10"

# (track key, human label, output root)
TRACKS: Final[tuple[tuple[str, str, Path], ...]] = (
    (
        "metadata_rag",
        "Metadata-RAG",
        REPO_ROOT / "outputs" / "external_metadata_rag",
    ),
    (
        "longrag",
        "LongRAG",
        REPO_ROOT / "outputs" / "external_longrag",
    ),
    (
        "my_rag",
        "My_RAG",
        REPO_ROOT / "outputs" / "external_my_rag",
    ),
)

COMPARISON_ARTIFACT_TYPE: Final[str] = (
    "metadata_rag_vs_baselines_comparison"
)
COMPARISON_ARTIFACT_VERSION: Final[str] = "1.0"

AGGREGATE_FILE: Final[str] = "aggregate_metrics.json"
LEGACY_AGGREGATE_FILE: Final[str] = "per_dataset_metrics.json"
AGGREGATE_FILE_CANDIDATES: Final[tuple[str, ...]] = (
    AGGREGATE_FILE,
    LEGACY_AGGREGATE_FILE,
)
EVALUATION_DIR: Final[str] = "evaluation-v1"


class ComparisonError(ValueError):
    """Raised when the comparison cannot be completed."""


# -------------------------------------------------------------------------
# I/O
# -------------------------------------------------------------------------


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _atomic_write_bytes(target: Path, payload: bytes) -> None:
    tmp = target.with_name(target.name + ".tmp")
    tmp.write_bytes(payload)
    tmp.replace(target)


def _write_json(path: Path, obj: Any) -> None:
    text = json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    _atomic_write_bytes(path, (text + "\n").encode("utf-8"))


def _read_json(path: Path) -> dict[str, Any]:
    parsed = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(parsed, dict):
        raise ComparisonError(f"{path.name} is not a JSON object")
    return parsed


# -------------------------------------------------------------------------
# Loading per-track metrics
# -------------------------------------------------------------------------


def _normalize_metric_keys(raw: dict[str, Any]) -> dict[str, float]:
    """Normalize legacy ``metric@k`` keys to native ``metric_at_k``.

    Native Metadata-RAG reports already use ``metric_at_k``. Legacy
    LongRAG and My_RAG reports use ``metric@k``. Both are normalized
    to the native form so the aggregation and the markdown renderer
    can operate on a single key vocabulary.
    """
    out: dict[str, float] = {}
    for key, value in raw.items():
        normalized = str(key).replace("@", "_at_")
        out[normalized] = float(value)
    return out


def _extract_mean_block(
    aggregate: dict[str, Any], source: Path
) -> dict[str, float]:
    """Extract the mean metric block from a native or legacy report.

    Native Metadata-RAG reports have a top-level ``mean`` mapping.
    Legacy LongRAG and My_RAG reports have a top-level
    ``macro_metrics`` mapping. Both are normalized to the native
    ``metric_at_k`` vocabulary.
    """
    mean_block = aggregate.get("mean")
    if isinstance(mean_block, dict):
        return _normalize_metric_keys(mean_block)
    macro_block = aggregate.get("macro_metrics")
    if isinstance(macro_block, dict):
        return _normalize_metric_keys(macro_block)
    raise ComparisonError(
        f"{source} has neither a 'mean' nor a 'macro_metrics' block"
    )


def _load_track_aggregates(
    root: Path,
    datasets: Iterable[str],
) -> dict[str, dict[str, Any]]:
    """Return {dataset: {"mean": normalized_metrics}} for one track.

    Supports both the native Metadata-RAG aggregate format
    (``aggregate_metrics.json`` with a ``mean`` block) and the legacy
    LongRAG / My_RAG aggregate format (``per_dataset_metrics.json``
    with a ``macro_metrics`` block). Missing datasets are silently
    absent.
    """
    out: dict[str, dict[str, Any]] = {}
    for ds in datasets:
        aggregate_path: Path | None = None
        for candidate in AGGREGATE_FILE_CANDIDATES:
            p = root / ds / EVALUATION_DIR / candidate
            if p.is_file():
                aggregate_path = p
                break
        if aggregate_path is None:
            continue
        aggregate = _read_json(aggregate_path)
        normalized = _extract_mean_block(aggregate, aggregate_path)
        out[ds] = {
            "dataset_id": aggregate.get("dataset_id", ds),
            "mean": normalized,
            "source": str(aggregate_path),
        }
    return out


# -------------------------------------------------------------------------
# Aggregation
# -------------------------------------------------------------------------


def _secondary_macro(
    per_dataset: dict[str, dict[str, Any]],
) -> dict[str, float]:
    """Macro-average over datasets of the per-dataset macro means."""
    if not per_dataset:
        return {}
    keys: set[str] = set()
    for aggregate in per_dataset.values():
        keys.update(aggregate["mean"].keys())
    out: dict[str, float] = {}
    n = len(per_dataset)
    for key in sorted(keys):
        total = 0.0
        for aggregate in per_dataset.values():
            total += float(aggregate["mean"].get(key, 0.0))
        out[key] = total / n
    return out


# -------------------------------------------------------------------------
# Comparison
# -------------------------------------------------------------------------


def build_comparison(
    *,
    tracks: tuple[tuple[str, str, Path], ...],
    datasets: tuple[str, ...],
) -> dict[str, Any]:
    per_track: dict[str, dict[str, Any]] = {}
    for key, label, root in tracks:
        aggregates = _load_track_aggregates(root, datasets)
        secondary = _secondary_macro(aggregates)
        per_track[key] = {
            "label": label,
            "root": str(root),
            "datasets_present": sorted(aggregates.keys()),
            "datasets_missing": sorted(
                set(datasets) - set(aggregates.keys())
            ),
            "per_dataset": {
                ds: aggregates[ds]["mean"]
                for ds in sorted(aggregates.keys())
            },
            "secondary_macro": secondary,
        }

    datasets_present_all = sorted(
        set(datasets)
        & set(
            ds
            for track in per_track.values()
            for ds in track["datasets_present"]
        )
    )

    comparison: dict[str, Any] = {
        "artifact_type": COMPARISON_ARTIFACT_TYPE,
        "artifact_version": COMPARISON_ARTIFACT_VERSION,
        "datasets_requested": list(datasets),
        "datasets_in_all_tracks": datasets_present_all,
        "cutoffs": list(CUTOFFS),
        "primary_metric": PRIMARY_METRIC,
        "tracks": per_track,
    }
    return comparison


# -------------------------------------------------------------------------
# Markdown rendering
# -------------------------------------------------------------------------


def render_markdown(comparison: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Metadata-RAG vs Baselines — Comparison\n")
    lines.append(
        "Retrieval-only comparison of `Metadata-RAG` against "
        "`LongRAG` and `My_RAG`.\n"
    )
    lines.append(
        f"**Primary metric:** macro document `{PRIMARY_METRIC}` "
        "across the four datasets.\n"
    )

    datasets = comparison["datasets_requested"]
    tracks = comparison["tracks"]

    lines.append("## 1. Per-dataset primary metric\n")
    header = "| Dataset | " + " | ".join(
        t["label"] for t in tracks.values()
    ) + " |"
    separator = "|" + "---|" * (len(tracks) + 1)
    lines.append(header)
    lines.append(separator)
    for ds in datasets:
        cells: list[str] = [ds]
        for track in tracks.values():
            value = track["per_dataset"].get(ds, {}).get(
                PRIMARY_METRIC
            )
            if value is None:
                cells.append("—")
            else:
                cells.append(f"{float(value):.4f}")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    lines.append("## 2. Secondary macro across datasets\n")
    header = "| Metric | " + " | ".join(
        t["label"] for t in tracks.values()
    ) + " |"
    separator = "|" + "---|" * (len(tracks) + 1)
    lines.append(header)
    lines.append(separator)
    for key, label in METRIC_KEYS:
        cells = [label]
        for track in tracks.values():
            value = track["secondary_macro"].get(key)
            if value is None:
                cells.append("—")
            else:
                cells.append(f"{float(value):.4f}")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    lines.append("## 3. Dataset coverage\n")
    lines.append("| Track | Present | Missing |")
    lines.append("|---|---|---|")
    for track in tracks.values():
        present = ", ".join(track["datasets_present"]) or "—"
        missing = ", ".join(track["datasets_missing"]) or "—"
        lines.append(
            f"| {track['label']} | {present} | {missing} |"
        )
    lines.append("")

    lines.append("## 4. Controlled adaptations\n")
    lines.append(
        "- All tracks are retrieval-only; generation is disabled.\n"
        "- Metadata-RAG uses the frozen clean-room spec plus "
        "amendments 01-04.\n"
        "- Native-fidelity track is DEFERRED_WITH_REASON for "
        "Metadata-RAG.\n"
    )
    return "\n".join(lines) + "\n"


# -------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare Metadata-RAG against LongRAG and My_RAG over "
            "the four datasets."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "baseline_audit" / "reports",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=list(EXPECTED_DATASETS),
    )
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)

    comparison = build_comparison(
        tracks=TRACKS,
        datasets=tuple(args.datasets),
    )

    if args.strict:
        for key, track in comparison["tracks"].items():
            if track["datasets_missing"]:
                raise SystemExit(
                    f"[strict] track {key} is missing datasets: "
                    f"{track['datasets_missing']}"
                )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "metadata_rag_vs_baselines.json"
    md_path = args.output_dir / "metadata_rag_vs_baselines.md"

    _write_json(json_path, comparison)
    _atomic_write_bytes(
        md_path, render_markdown(comparison).encode("utf-8")
    )

    print("[OK] comparison report written")
    print(f"  json: {json_path}")
    print(f"  md:   {md_path}")

    print("\nsecondary macro primary metric:")
    for _key, track in comparison["tracks"].items():
        value = track["secondary_macro"].get(PRIMARY_METRIC)
        value_str = f"{float(value):.4f}" if value is not None else "n/a"
        print(
            f"  {track['label']:<16} {value_str} "
            f"(datasets: {len(track['datasets_present'])})"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
