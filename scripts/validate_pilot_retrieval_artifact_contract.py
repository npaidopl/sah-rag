"""Validate the Step 86 retrieval artifact contract."""

from __future__ import annotations

from pathlib import Path

import yaml

PATH = Path("baseline_audit/pilot_retrieval_artifact_contract.yaml")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def main() -> int:
    data = yaml.safe_load(PATH.read_text(encoding="utf-8"))
    _require(isinstance(data, dict), "artifact contract must be a mapping")
    _require(
        data.get("artifact_type") == "pilot_retrieval_artifact_contract",
        "artifact_type changed",
    )

    contract = data["contract"]
    _require(
        contract["id"] == "infrastructure_pilot_ranked_retrieval_artifacts",
        "contract.id changed",
    )
    _require(contract["version"] == "0.1.0", "contract.version must be 0.1.0")

    rc = data["ranked_children"]
    _require(
        rc["retrieval_methods"] == ["dense", "sparse", "hybrid_rrf"],
        "ranked_children.retrieval_methods changed",
    )
    _require(rc["constraints"]["rank_starts_at_one"] is True, "rank_starts_at_one must be true")
    _require(
        rc["constraints"]["ranks_contiguous_per_query_and_method"] is True,
        "ranks_contiguous must be true",
    )
    _require(
        rc["constraints"]["score_direction"] == "HIGHER_IS_BETTER",
        "score_direction must be HIGHER_IS_BETTER",
    )

    rp = data["ranked_parents"]
    for name in (
        "query_id",
        "parent_id",
        "doc_id",
        "rank",
        "score",
        "contributing_chunk_ids",
        "metadata",
    ):
        _require(name in rp["required_fields"], "ranked_parents missing " + name)

    rd = data["ranked_documents"]
    _require(
        rd["constraints"]["duplicate_content_documents_not_deduplicated"] is True,
        "duplicate_content_documents_not_deduplicated must be true",
    )
    for name in (
        "query_id",
        "doc_id",
        "rank",
        "score",
        "contributing_parent_ids",
        "contributing_chunk_ids",
        "metadata",
    ):
        _require(name in rd["required_fields"], "ranked_documents missing " + name)

    ser = data["serialization"]
    _require(ser["encoding"] == "UTF-8", "encoding must be UTF-8")
    _require(ser["line_endings"] == "LF", "line_endings must be LF")
    _require(ser["checksum_algorithm"] == "SHA-256", "checksum_algorithm must be SHA-256")
    _require(ser["atomic_writes_required"] is True, "atomic_writes_required must be true")

    prohibited = set(data["prohibited_fields"])
    for name in (
        "generated_answer",
        "reference_answer",
        "relevance",
        "judgment",
        "evaluation_metric",
    ):
        _require(name in prohibited, "prohibited_fields missing " + name)

    # no evaluation/answer fields in required fields of any ranked item
    for block in (rc, rp, rd):
        for f in block["required_fields"]:
            _require(
                f not in prohibited,
                "prohibited field " + f + " present in required_fields",
            )

    print("[OK] Pilot retrieval artifact contract")
    print("  query embeddings: planned")
    print("  ranked child methods:", rc["retrieval_methods"])
    print("  ranked parents: planned")
    print("  ranked documents: planned")
    print("  score direction:", rc["constraints"]["score_direction"])
    print("  evaluation fields prohibited: True")
    print("  answer fields prohibited: True")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())