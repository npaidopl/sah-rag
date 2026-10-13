"""Deterministic planner for the Step 89 bounded retrieval evaluation.

Validates the Step 87 retrieval execution manifest, loads the six
ranked-document query groups, inspects the pilot judgments for
validation only, verifies the frozen common evaluation contract,
and computes the judgment-plan and evaluation-plan hashes. Never
calls the evaluator, never creates an evaluation output directory,
never computes metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]

EVALUATION_CONTRACT = REPO_ROOT / "baseline_audit" / "pilot_evaluation_contract.yaml"
EVALUATION_MANIFEST = REPO_ROOT / "baseline_audit" / "pilot_evaluation_manifest.yaml"
JUDGMENT_CONTRACT = REPO_ROOT / "baseline_audit" / "pilot_judgment_mapping_contract.yaml"
ARTIFACT_CONTRACT = REPO_ROOT / "baseline_audit" / "pilot_evaluation_artifact_contract.yaml"
COMMON_EVAL_CONTRACT = REPO_ROOT / "baseline_audit" / "common_retrieval_evaluation_contract.yaml"
RETRIEVAL_EXECUTION_MANIFEST = (
    REPO_ROOT / "baseline_audit" / "pilot_retrieval_execution_manifest.yaml"
)

FROZEN_DATASET = "infrastructure-pilot-synthetic-v1"
FROZEN_RETRIEVAL_PLAN_HASH = (
    "950fad1a871d8440f6f8916adf67d59f019566924053e8cdd33f4d5108dfe7e6"
)

EXPECTED_CUTOFFS = [1, 5, 10, 100]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit("[FAIL] " + message)


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _canonical_sha256(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def validate_dependencies() -> dict[str, Any]:
    ec = _load_yaml(EVALUATION_CONTRACT)
    _require(
        ec["contract"]["status"] == "FROZEN_FOR_PLANNING",
        "evaluation contract must be FROZEN_FOR_PLANNING",
    )

    rem = _load_yaml(RETRIEVAL_EXECUTION_MANIFEST)
    _require(
        rem["execution"]["status"] == "SUCCEEDED",
        "retrieval execution manifest must be SUCCEEDED",
    )
    _require(
        rem["results"]["ranked_document_count"] == 72,
        "retrieval ranked_document_count must be 72",
    )
    _require(
        rem["results"]["judgments_loaded"] is False,
        "retrieval must not have loaded judgments",
    )
    _require(
        rem["results"]["evaluation_executed"] is False,
        "retrieval must not have executed evaluation",
    )

    cec = _load_yaml(COMMON_EVAL_CONTRACT)
    _require(
        cec["contract"]["version"] == "0.1.0",
        "common evaluation contract version must be 0.1.0",
    )
    _require(
        cec["safety"]["retrieval_only"] is True,
        "common contract must be retrieval_only",
    )
    _require(
        cec["safety"]["final_generation_enabled"] is False,
        "common contract must not enable final generation",
    )
    _require(
        cec["safety"]["answer_evaluation_enabled"] is False,
        "common contract must not enable answer evaluation",
    )
    _require(
        list(cec["cutoffs"]) == EXPECTED_CUTOFFS,
        "common contract cutoffs must be [1, 5, 10, 100]",
    )

    _ = _load_yaml(JUDGMENT_CONTRACT)
    _ = _load_yaml(ARTIFACT_CONTRACT)

    return ec


def load_ranked_documents_grouped(
    ec: dict[str, Any],
) -> dict[str, list[str]]:
    path = REPO_ROOT / ec["source"]["ranked_documents_path"]
    _require(path.exists(), "ranked_documents.jsonl missing")
    records = _load_jsonl(path)

    grouped: dict[str, list[tuple[int, str]]] = {}
    for r in records:
        grouped.setdefault(str(r["query_id"]), []).append(
            (int(r["rank"]), str(r["doc_id"]))
        )

    result: dict[str, list[str]] = {}
    for qid, items in grouped.items():
        items.sort(key=lambda t: t[0])
        result[qid] = [doc_id for _, doc_id in items]

    expected = int(ec["source"]["expected_query_count"])
    _require(len(result) == expected, "ranked query groups must be " + str(expected))
    return result


def load_judgments_grouped(
    ec: dict[str, Any],
) -> dict[str, list[str]]:
    path = REPO_ROOT / ec["source"]["judgments_path"]
    _require(path.exists(), "judgments.jsonl missing")
    records = _load_jsonl(path)

    grouped: dict[str, list[str]] = {}
    for r in records:
        _require(int(r["relevance"]) == 1, "relevance must be 1 for relevant items")
        grouped.setdefault(str(r["query_id"]), []).append(str(r["item_id"]))

    for qid in grouped:
        grouped[qid] = sorted(grouped[qid])

    expected_q = int(ec["source"]["expected_query_count"])
    expected_j = int(ec["source"]["expected_judgment_count"])
    per_q = int(ec["source"]["expected_relevant_documents_per_query"])

    _require(len(grouped) == expected_q, "judgment query count must be " + str(expected_q))
    total = sum(len(v) for v in grouped.values())
    _require(total == expected_j, "judgment count must be " + str(expected_j))
    for qid, docs in grouped.items():
        _require(len(docs) == per_q, qid + ": must have " + str(per_q) + " relevant docs")
        _require(len(set(docs)) == len(docs), qid + ": duplicate relevant docs")

    return grouped


def verify_document_identity(
    ranked: dict[str, list[str]],
    judgments: dict[str, list[str]],
    canonical_doc_ids: set[str],
) -> None:
    for qid, docs in ranked.items():
        _require(qid in judgments, qid + ": missing judgments")
        for d in docs:
            _require(d in canonical_doc_ids, qid + ": unknown doc_id " + d)
    for qid, docs in judgments.items():
        for d in docs:
            _require(
                d in canonical_doc_ids,
                qid + ": judgment references unknown canonical doc " + d,
            )


def load_canonical_doc_ids() -> set[str]:
    path = REPO_ROOT / "processed/canonical/infrastructure-pilot-synthetic-v1/documents.jsonl"
    _require(path.exists(), "canonical documents.jsonl missing")
    records = _load_jsonl(path)
    ids = {str(r["doc_id"]) for r in records}
    _require(len(ids) == 12, "must be exactly 12 canonical documents")
    return ids


def compute_judgment_plan_hash(
    judgments: dict[str, list[str]],
    query_ids: list[str],
) -> str:
    identity = {
        "dataset_id": FROZEN_DATASET,
        "split": "pilot",
        "query_ids": sorted(query_ids),
        "relevant_doc_ids": {
            qid: sorted(judgments[qid]) for qid in sorted(judgments)
        },
        "relevance_type": "binary",
        "relevant_value": 1,
    }
    return _canonical_sha256(identity)


def compute_evaluation_plan_hash(
    judgment_plan_hash: str,
    metric_names: list[str],
) -> str:
    identity = {
        "judgment_plan_hash": judgment_plan_hash,
        "retrieval_plan_hash": FROZEN_RETRIEVAL_PLAN_HASH,
        "retrieval_method": "hybrid_rrf",
        "evaluation_unit": "canonical_document",
        "cutoffs": list(EXPECTED_CUTOFFS),
        "metric_names": sorted(metric_names),
        "short_result_policy": "evaluate_available_prefix_or_all",
        "empty_judgment_policy": "reject_case",
        "duplicate_content_deduplication": False,
    }
    return _canonical_sha256(identity)


def update_planning_manifest(
    metric_names: list[str],
    judgment_plan_hash: str,
    evaluation_plan_hash: str,
) -> None:
    manifest = _load_yaml(EVALUATION_MANIFEST)
    _require(
        manifest["execution"]["status"] == "PLANNED",
        "manifest must start as PLANNED",
    )
    manifest["plan"]["metric_names"] = list(metric_names)
    manifest["plan"]["judgment_plan_hash"] = judgment_plan_hash
    manifest["plan"]["evaluation_plan_hash"] = evaluation_plan_hash
    EVALUATION_MANIFEST.write_text(
        yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-update-manifest",
        action="store_true",
        help="Preview only; do not write the planning manifest.",
    )
    args = parser.parse_args()

    ec = validate_dependencies()

    ranked = load_ranked_documents_grouped(ec)
    judgments = load_judgments_grouped(ec)
    canonical_ids = load_canonical_doc_ids()
    verify_document_identity(ranked, judgments, canonical_ids)

    metric_names = sorted(ec["metric_names"])
    query_ids = sorted(ranked.keys())

    judgment_plan_hash = compute_judgment_plan_hash(judgments, query_ids)
    evaluation_plan_hash = compute_evaluation_plan_hash(
        judgment_plan_hash, metric_names
    )

    art = _load_yaml(ARTIFACT_CONTRACT)
    evaluation_root = REPO_ROOT / art["outputs"]["root"]
    _require(
        not evaluation_root.exists(),
        "evaluation output root already exists: " + str(evaluation_root),
    )

    if not args.no_update_manifest:
        update_planning_manifest(
            metric_names, judgment_plan_hash, evaluation_plan_hash
        )

    print("[OK] Pilot retrieval evaluation plan")
    print("  dataset:", FROZEN_DATASET)
    print("  retrieval method: hybrid_rrf")
    print()
    print("  queries:", len(query_ids))
    print("  ranked-document query groups:", len(ranked))
    print("  judgments:", sum(len(v) for v in judgments.values()))
    print("  relevant documents per query:", 2)
    print()
    print("  evaluation unit: canonical_document")
    print("  duplicate document identities preserved: True")
    print("  content deduplication: False")
    print()
    print("  cutoffs:", EXPECTED_CUTOFFS)
    print("  metrics:", ", ".join(metric_names))
    print("  short-result policy: evaluate_available_prefix_or_all")
    print("  empty-judgment policy: reject_case")
    print()
    print("  judgment plan hash:", judgment_plan_hash)
    print("  evaluation plan hash:", evaluation_plan_hash)
    print()
    print("  planning judgments inspected: True")
    print("  runtime judgments loaded: False")
    print("  evaluation executed: False")
    print("  metrics computed: False")
    print("  generation executed: False")
    print("  output directory created: False")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())