"""Validate the LongRAG readiness contract."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

READINESS_PATH = Path("baseline_audit/longrag_readiness.yaml")

EXPECTED_FLAGS = {
    "adapter_ready": True,
    "data_ready": True,
    "execution_ready": True,
    "full_embeddings_generated": False,
    "production_qdrant_collections_created": False,
    "real_retrieval_executed": False,
    "real_evaluation_executed": False,
}

EXPECTED_EVIDENCE: dict[str, tuple[str, ...]] = {
    "adapter": (
        "src/sah_rag/baselines/longrag.py",
        "src/sah_rag/baselines/longrag_embeddings.py",
        "schemas/longrag_long_unit.schema.json",
        "schemas/longrag_embedding_record.schema.json",
        "schemas/longrag_embedding_shard_header.schema.json",
        "schemas/longrag_embedding_catalog.schema.json",
    ),
    "data": (
        "outputs/longrag_clerk_source_corrected_pilot/LONGRAG-SD-4096/units.jsonl",
        "outputs/longrag_clerk_source_corrected_pilot/LONGRAG-ADJ-2048/units.jsonl",
        "outputs/longrag_clerk_source_corrected_pilot/LONGRAG-ADJ-4096/units.jsonl",
    ),
    "execution": (
        "src/sah_rag/baselines/longrag_storage.py",
        "src/sah_rag/baselines/longrag_retrieval.py",
        "schemas/longrag_collection_manifest.schema.json",
        "schemas/longrag_retrieved_unit.schema.json",
    ),
    "smoke": (
        "scripts/smoke_longrag_dense_retrieval.py",
        "tests/unit/test_longrag_storage.py",
        "tests/unit/test_longrag_retrieval.py",
    ),
}


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected YAML mapping: {path}")
    return value


def main() -> None:
    readiness = load_yaml(READINESS_PATH)
    longrag = readiness["longrag"]

    for key, expected in EXPECTED_FLAGS.items():
        actual = longrag.get(key)
        if actual is not expected:
            raise RuntimeError(
                f"LongRAG readiness flag {key!r}: "
                f"expected {expected!r}, got {actual!r}."
            )

    smoke = longrag["synthetic_end_to_end_smoke"]
    if smoke != "passed":
        raise RuntimeError(
            f"synthetic_end_to_end_smoke must be 'passed', got {smoke!r}."
        )

    status = longrag["readiness_status"]
    if status != "READY_FOR_DATA":
        raise RuntimeError(
            f"readiness_status must be READY_FOR_DATA, got {status!r}."
        )

    if readiness["foreign_code_execution_performed"] is not False:
        raise RuntimeError("foreign_code_execution_performed must be false.")
    if readiness["bge_m3_inference_performed_in_smoke"] is not False:
        raise RuntimeError("bge_m3_inference_performed_in_smoke must be false.")
    if readiness["production_qdrant_collections_created"] is not False:
        raise RuntimeError(
            "production_qdrant_collections_created must be false."
        )

    evidence = readiness["evidence"]
    for group, files in EXPECTED_EVIDENCE.items():
        if group not in evidence:
            raise RuntimeError(f"Missing evidence group: {group}")
        recorded = set(evidence[group])
        for path in files:
            if path not in recorded:
                raise RuntimeError(
                    f"Missing evidence path in {group}: {path}"
                )
            if not Path(path).is_file():
                raise RuntimeError(f"Evidence path does not exist: {path}")

    print("[OK] LongRAG readiness")
    print("  adapter_ready:      ", longrag["adapter_ready"])
    print("  data_ready:         ", longrag["data_ready"])
    print("  execution_ready:    ", longrag["execution_ready"])
    print("  smoke:              ", longrag["synthetic_end_to_end_smoke"])
    print("  embeddings_generated:", longrag["full_embeddings_generated"])
    print("  collections_created: ", longrag["production_qdrant_collections_created"])
    print("  retrieval_executed:  ", longrag["real_retrieval_executed"])
    print("  evaluation_executed: ", longrag["real_evaluation_executed"])
    print("  readiness_status:   ", longrag["readiness_status"])


if __name__ == "__main__":
    main()
