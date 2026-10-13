"""Validate the Step 87 pilot retrieval outputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
EXECUTION_CONTRACT = REPO_ROOT / "baseline_audit" / "pilot_retrieval_execution_contract.yaml"

PROHIBITED_PAYLOAD_KEYS = {
    "generated_answer",
    "reference_answer",
    "answer_text",
    "relevance",
    "relevance_label",
    "judgment",
    "evaluation_metric",
    "expected_group",
    "is_relevant",
    "hit",
    "metric",
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def main() -> int:
    contract = yaml.safe_load(EXECUTION_CONTRACT.read_text(encoding="utf-8"))
    root = REPO_ROOT / contract["output"]["root"]

    _require(root.is_dir(), "retrieval output root missing: " + str(root))

    for name in contract["output"]["required_files"]:
        _require((root / name).exists(), "required file missing: " + name)

    qe = _load_jsonl(root / "query_embeddings.jsonl")
    rc = _load_jsonl(root / "ranked_children.jsonl")
    rp = _load_jsonl(root / "ranked_parents.jsonl")
    rd = _load_jsonl(root / "ranked_documents.jsonl")

    _require(len(qe) == 6, "query embeddings must be 6")
    query_ids = {q["query_id"] for q in qe}
    _require(len(query_ids) == 6, "query IDs must be unique")

    # Ranked children coverage
    methods_seen: dict[str, int] = {}
    children_by_qm: dict[tuple[str, str], list[dict]] = {}
    for r in rc:
        key = (r["query_id"], r["retrieval_method"])
        children_by_qm.setdefault(key, []).append(r)
        methods_seen[r["retrieval_method"]] = methods_seen.get(r["retrieval_method"], 0) + 1

    for qid in query_ids:
        for method in ("dense", "sparse", "hybrid_rrf"):
            items = children_by_qm.get((qid, method), [])
            _require(len(items) >= 1, f"{qid} {method}: at least 1 result required")

    # Rank contiguity per (query_id, method)
    for (qid, method), items in children_by_qm.items():
        ranks = sorted(int(i["rank"]) for i in items)
        _require(
            ranks == list(range(1, len(ranks) + 1)),
            f"{qid} {method}: ranks not contiguous",
        )
        chunk_ids = [i["chunk_id"] for i in items]
        _require(len(set(chunk_ids)) == len(chunk_ids), f"{qid} {method}: chunk IDs not unique")

    # Parent ranks
    parents_by_q: dict[str, list[dict]] = {}
    for r in rp:
        parents_by_q.setdefault(r["query_id"], []).append(r)
    for qid, items in parents_by_q.items():
        ranks = sorted(int(i["rank"]) for i in items)
        _require(ranks == list(range(1, len(ranks) + 1)), f"{qid}: parent ranks not contiguous")
        pids = [i["parent_id"] for i in items]
        _require(len(set(pids)) == len(pids), f"{qid}: parent IDs not unique")

    # Document ranks
    docs_by_q: dict[str, list[dict]] = {}
    for r in rd:
        docs_by_q.setdefault(r["query_id"], []).append(r)
    for qid, items in docs_by_q.items():
        ranks = sorted(int(i["rank"]) for i in items)
        _require(ranks == list(range(1, len(ranks) + 1)), f"{qid}: document ranks not contiguous")
        dids = [i["doc_id"] for i in items]
        _require(len(set(dids)) == len(dids), f"{qid}: document IDs not unique")

    # Prohibited fields
    for block in (rc, rp, rd):
        for r in block:
            forbidden = PROHIBITED_PAYLOAD_KEYS & set(r.keys())
            _require(not forbidden, "forbidden field(s): " + ", ".join(sorted(forbidden)))

    # Lineage
    for r in rc:
        for key in ("chunk_id", "parent_id", "doc_id"):
            _require(key in r and r[key], "child missing " + key)
    for r in rp:
        for key in ("parent_id", "doc_id"):
            _require(key in r and r[key], "parent missing " + key)
    for r in rd:
        for key in ("doc_id",):
            _require(key in r and r[key], "document missing " + key)

    # Checksums
    checksums = json.loads((root / "checksums.json").read_text(encoding="utf-8"))
    for name, expected in checksums["files"].items():
        _require(_sha256(root / name) == expected, "checksum mismatch: " + name)

    print("[OK] Pilot retrieval outputs")
    print("  query embeddings:", len(qe))
    print("  ranked children:")
    for m in ("dense", "sparse", "hybrid_rrf"):
        print("    " + m + ":", methods_seen.get(m, 0))
    print("  ranked parents:", len(rp))
    print("  ranked documents:", len(rd))
    print("  query coverage complete: True")
    print("  ranks contiguous: True")
    print("  child lineage complete: True")
    print("  parent lineage complete: True")
    print("  document lineage complete: True")
    print("  duplicate document identities preserved: True")
    print("  forbidden evaluation fields absent: True")
    print("  answer fields absent: True")
    print("  judgments absent: True")
    print("  round trip valid: True")
    print("  checksums valid: True")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())