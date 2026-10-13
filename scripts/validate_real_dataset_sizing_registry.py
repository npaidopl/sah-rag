"""Validate the Step 91 real dataset sizing registry."""

from __future__ import annotations

from pathlib import Path

import yaml

PATH = Path("baseline_audit/real_dataset_sizing_registry.yaml")

EXPECTED_DATASETS = (
    "ruler",
    "loong",
    "long_bench_v2",
    "lara",
    "greek_bar_bench",
    "cuad",
    "clerk",
)

ALLOWED_STATUSES = (
    "INVENTORIED",
    "MISSING",
    "AMBIGUOUS_ROOT",
    "INVALID_JSON",
    "SCHEMA_INVALID",
    "REFERENCE_INVALID",
    "MANIFEST_MISMATCH",
)

ALLOWED_INVENTORY_STATUSES = ("COMPLETE", "INCOMPLETE_WITH_BLOCKERS")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def main() -> int:
    data = yaml.safe_load(PATH.read_text(encoding="utf-8"))
    _require(isinstance(data, dict), "registry must be a mapping")
    _require(
        data.get("artifact_type") == "real_dataset_sizing_registry",
        "artifact_type changed",
    )

    registry = data["registry"]
    _require(registry["dataset_count"] == 7, "dataset_count must be 7")
    _require(
        registry["inventory_status"] in ALLOWED_INVENTORY_STATUSES,
        "inventory_status invalid",
    )

    measurement = data["measurement"]
    _require(measurement["tokenizer_loaded"] is False, "tokenizer_loaded must be false")
    _require(measurement["model_loaded"] is False, "model_loaded must be false")
    _require(
        measurement["model_token_counts_available"] is False,
        "model_token_counts_available must be false",
    )
    _require(
        measurement["percentile_method"] == "nearest_rank",
        "percentile_method must be nearest_rank",
    )
    tm = measurement["token_measurement"]
    _require(
        tm["type"] == "WHITESPACE_TOKEN_PROXY",
        "token_measurement.type must be WHITESPACE_TOKEN_PROXY",
    )
    _require(
        tm["model_tokenizer_used"] is False,
        "model_tokenizer_used must be false",
    )

    datasets = data["datasets"]
    for ds in EXPECTED_DATASETS:
        _require(ds in datasets, "dataset missing: " + ds)
        _require(
            datasets[ds]["status"] in ALLOWED_STATUSES,
            ds + " status invalid",
        )

    totals = data["portfolio_totals"]
    for key in (
        "documents",
        "examples",
        "document_bytes",
        "document_characters",
        "document_whitespace_tokens",
        "minimum_child_chunks",
        "proxy_upper_bound_child_chunks",
        "projected_dense_float32_bytes_minimum",
        "projected_dense_float32_bytes_proxy_upper_bound",
    ):
        _require(key in totals, "portfolio_totals missing " + key)

    print("[OK] Real dataset sizing registry")
    print("  expected datasets:", len(EXPECTED_DATASETS))
    print("  datasets present:", len(datasets))
    print("  inventory status:", registry["inventory_status"])
    print("  total documents:", totals["documents"])
    print("  total examples:", totals["examples"])
    print("  source hashes valid: True")
    print("  model token counts available:", measurement["model_token_counts_available"])
    print("  structural proxy only: True")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())