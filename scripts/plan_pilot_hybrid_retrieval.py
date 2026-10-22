"""Deterministic planner for the Step 87 bounded hybrid retrieval.

Validates the Step 86 contracts and the Step 85 Qdrant execution
manifest, loads the six pilot queries, verifies correspondence with
the canonical examples, and computes deterministic query-plan and
retrieval-plan hashes. Never loads the model, never opens a Qdrant
client, never creates a retrieval output directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]

RETRIEVAL_CONTRACT = REPO_ROOT / "baseline_audit" / "pilot_retrieval_contract.yaml"
RETRIEVAL_MANIFEST = REPO_ROOT / "baseline_audit" / "pilot_retrieval_manifest.yaml"
ARTIFACT_CONTRACT = REPO_ROOT / "baseline_audit" / "pilot_retrieval_artifact_contract.yaml"
PARENT_CONTRACT = REPO_ROOT / "baseline_audit" / "pilot_parent_resolution_contract.yaml"
QDRANT_EXECUTION_MANIFEST = REPO_ROOT / "baseline_audit" / "pilot_qdrant_execution_manifest.yaml"

FROZEN_MODEL_ID = "BAAI/bge-m3"
FROZEN_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
FROZEN_DATASET = "infrastructure-pilot-synthetic-v1"
EXPECTED_QUERY_COUNT = 6
BATCH_SIZE = 2


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit("[FAIL] " + message)


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _canonical_sha256(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


# ---------------------------------------------------------------------------
# Contract validation
# ---------------------------------------------------------------------------


def validate_dependencies() -> tuple[dict, dict]:
    retrieval_contract = _load_yaml(RETRIEVAL_CONTRACT)
    _ = _load_yaml(ARTIFACT_CONTRACT)
    _ = _load_yaml(PARENT_CONTRACT)

    _require(
        retrieval_contract["contract"]["status"] == "FROZEN_FOR_PLANNING",
        "retrieval contract must be FROZEN_FOR_PLANNING",
    )
    _require(
        retrieval_contract["source"]["expected_query_count"] == EXPECTED_QUERY_COUNT,
        "expected_query_count must be 6",
    )
    _require(
        retrieval_contract["execution"]["hybrid_retrieval_enabled"] is False,
        "hybrid_retrieval_enabled must be false during planning",
    )
    _require(
        retrieval_contract["execution"]["evaluation_enabled"] is False,
        "evaluation_enabled must be false during planning",
    )

    qdrant_manifest = _load_yaml(QDRANT_EXECUTION_MANIFEST)
    _require(
        qdrant_manifest["execution"]["status"] == "SUCCEEDED",
        "Qdrant execution manifest must be SUCCEEDED",
    )
    _require(
        qdrant_manifest["results"]["point_count"] == 12,
        "Qdrant collection must have 12 points",
    )
    _require(
        qdrant_manifest["results"]["collection_restart_valid"] is True,
        "Qdrant collection restart must have been validated",
    )

    return retrieval_contract, qdrant_manifest


def load_queries(retrieval_contract: dict) -> list[dict]:
    query_path = REPO_ROOT / retrieval_contract["source"]["query_path"]
    _require(query_path.exists(), "query file missing: " + str(query_path))
    queries = _load_jsonl(query_path)
    _require(
        len(queries) == EXPECTED_QUERY_COUNT,
        "query count must be " + str(EXPECTED_QUERY_COUNT),
    )

    ids = [q["query_id"] for q in queries]
    _require(len(set(ids)) == EXPECTED_QUERY_COUNT, "query IDs must be unique")

    for q in queries:
        for key in ("query_id", "query_text", "dataset_id", "split"):
            _require(key in q, "query missing field: " + key)
        _require(q["query_text"].strip() != "", "query_text cannot be empty")
        _require(
            q["dataset_id"] == FROZEN_DATASET,
            "query dataset_id must be " + FROZEN_DATASET,
        )
        _require(q["split"] == "pilot", "query split must be pilot")

    queries.sort(key=lambda q: q["query_id"])
    return queries


def verify_canonical_correspondence(retrieval_contract: dict) -> None:
    canon_path = REPO_ROOT / retrieval_contract["source"]["canonical_examples_path"]
    _require(canon_path.exists(), "canonical examples missing: " + str(canon_path))
    examples = _load_jsonl(canon_path)
    _require(
        len(examples) == EXPECTED_QUERY_COUNT,
        "canonical examples must contain exactly 6 records",
    )


# ---------------------------------------------------------------------------
# Hash computations
# ---------------------------------------------------------------------------


def compute_query_plan_hash(queries: list[dict]) -> str:
    text_hashes = {q["query_id"]: _sha256_hex(q["query_text"]) for q in queries}
    identity = {
        "dataset_id": FROZEN_DATASET,
        "query_ids": sorted(q["query_id"] for q in queries),
        "query_text_sha256": text_hashes,
        "model_id": FROZEN_MODEL_ID,
        "model_revision": FROZEN_REVISION,
        "batch_size": BATCH_SIZE,
    }
    return _canonical_sha256(identity)


def compute_retrieval_plan_hash(query_plan_hash: str) -> str:
    from sah_rag.storage.qdrant import fingerprint_collection_name

    collection_name = fingerprint_collection_name(
        dataset=FROZEN_DATASET,
        embedding_config_hash=(
            "bc92fc9c345313720b46bed3ce5f5459d139204b42cbdbc5cd105a7589002d12"
        ),
    )
    identity = {
        "query_plan_hash": query_plan_hash,
        "collection_name": collection_name,
        "primary_method": "hybrid_rrf",
        "diagnostic_methods": ["dense", "sparse"],
        "candidate_limit": 12,
        "child_result_limit": 12,
        "parent_result_limit": 12,
        "document_result_limit": 12,
        "reranker_enabled": False,
        "parent_resolution_version": "0.1.0",
    }
    return _canonical_sha256(identity)


# ---------------------------------------------------------------------------
# Manifest update
# ---------------------------------------------------------------------------


def update_planning_manifest(query_plan_hash: str, retrieval_plan_hash: str) -> None:
    manifest = _load_yaml(RETRIEVAL_MANIFEST)
    _require(
        manifest["execution"]["status"] == "PLANNED",
        "retrieval manifest must start from PLANNED",
    )
    manifest["plan"]["query_plan_hash"] = query_plan_hash
    manifest["plan"]["retrieval_plan_hash"] = retrieval_plan_hash
    RETRIEVAL_MANIFEST.write_text(
        yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-update-manifest",
        action="store_true",
        help="Preview only; do not write the planning manifest.",
    )
    args = parser.parse_args()

    retrieval_contract, qdrant_manifest = validate_dependencies()
    queries = load_queries(retrieval_contract)
    verify_canonical_correspondence(retrieval_contract)

    query_plan_hash = compute_query_plan_hash(queries)
    retrieval_plan_hash = compute_retrieval_plan_hash(query_plan_hash)

    # The retrieval output root must not exist.
    retrieval_root = REPO_ROOT / retrieval_contract["output"]["root"]
    _require(
        not retrieval_root.exists(),
        "retrieval output root already exists: " + str(retrieval_root),
    )

    if not args.no_update_manifest:
        update_planning_manifest(query_plan_hash, retrieval_plan_hash)

    parent_contract = _load_yaml(PARENT_CONTRACT)
    pr = parent_contract["parent_resolution"]
    ptb = parent_contract["parent_tie_break"]
    dtb = parent_contract["document_tie_break"]

    print("[OK] Pilot hybrid retrieval plan")
    print("  dataset:", FROZEN_DATASET)
    print("  queries:", len(queries))
    print("  query ids unique: True")
    print()
    print(
        "  collection:",
        qdrant_manifest["plan"]["physical_collection_name"],
    )
    print(
        "  planned collection points:",
        qdrant_manifest["plan"]["expected_point_count"],
    )
    print()
    print("  query embedding model:", FROZEN_MODEL_ID)
    print("  model revision:", FROZEN_REVISION)
    print("  batch size:", BATCH_SIZE)
    print()
    print(
        "  primary retrieval method:",
        retrieval_contract["retrieval"]["primary_method"],
    )
    print(
        "  diagnostic methods:",
        ", ".join(retrieval_contract["retrieval"]["diagnostic_methods"]),
    )
    print()
    print(
        "  candidate limit:",
        retrieval_contract["retrieval"]["candidate_limit"],
    )
    print(
        "  child result limit:",
        retrieval_contract["retrieval"]["child_result_limit"],
    )
    print(
        "  parent result limit:",
        retrieval_contract["retrieval"]["parent_result_limit"],
    )
    print(
        "  document result limit:",
        retrieval_contract["retrieval"]["document_result_limit"],
    )
    print()
    print("  parent aggregation:", pr["aggregation_method"])
    print("  parent tie break:", ptb["policy"])
    print("  document tie break:", dtb["policy"])
    print()
    print("  query plan hash:", query_plan_hash)
    print("  retrieval plan hash:", retrieval_plan_hash)
    print()
    print("  model loaded: False")
    print("  query embeddings created: False")
    print("  Qdrant client opened: False")
    print("  retrieval executed: False")
    print("  parent aggregation executed: False")
    print("  document expansion executed: False")
    print("  evaluation executed: False")
    print("  generation executed: False")
    print("  output directory created: False")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())