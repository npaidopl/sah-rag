"""Validate the output tree produced by run_longrag_synthetic_acceptance.py."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, cast

ROOT = Path(
    "outputs/infrastructure_pilot/infrastructure-pilot-synthetic-v1/longrag"
)

REQUIRED_FILES = [
    ROOT / "units" / "long_units.jsonl",
    ROOT / "units" / "unit.manifest.json",
    ROOT / "units" / "checksums.json",
    ROOT / "embeddings" / "embedding_records.jsonl",
    ROOT / "embeddings" / "embedding.manifest.json",
    ROOT / "embeddings" / "checksums.json",
    ROOT / "reference_index" / "index.manifest.json",
    ROOT / "reference_index" / "checksums.json",
    ROOT / "retrieval" / "reference_rankings.jsonl",
    ROOT / "retrieval" / "qdrant_rankings.jsonl",
    ROOT / "retrieval" / "ranked_documents.jsonl",
    ROOT / "retrieval" / "retrieval.manifest.json",
    ROOT / "retrieval" / "checksums.json",
    ROOT / "evaluation_cases" / "retrieval_cases.jsonl",
    ROOT / "evaluation_cases" / "evaluation_case.manifest.json",
    ROOT / "evaluation_cases" / "checksums.json",
]


def fail(msg: str) -> None:
    print(f"[FAIL] {msg}")
    sys.exit(1)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    rows: list[dict[str, Any]] = []
    for line in lines:
        if line.strip():
            parsed = json.loads(line)
            if not isinstance(parsed, dict):
                fail(f"{path}: expected JSON object per line")
            rows.append(parsed)
    return rows


def read_json_object(path: Path) -> dict[str, Any]:
    parsed = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(parsed, dict):
        fail(f"{path}: expected a top-level JSON object")
    return cast("dict[str, Any]", parsed)


def verify_checksums(checksum_path: Path, base: Path) -> None:
    d = read_json_object(checksum_path)
    files_map_raw = d.get("files")
    if not isinstance(files_map_raw, dict):
        fail(f"{checksum_path}: missing or invalid 'files' key")
    files_map = cast("dict[str, Any]", files_map_raw)
    for rel, expected in files_map.items():
        if not isinstance(rel, str) or not isinstance(expected, str):
            fail(f"{checksum_path}: invalid entry {rel!r}")
        target = base / rel
        if not target.exists():
            fail(f"checksum references missing file: {target}")
        actual = sha256_file(target)
        if actual != expected:
            fail(f"checksum mismatch for {target}: {actual} != {expected}")


def main() -> None:
    for p in REQUIRED_FILES:
        if not p.exists():
            fail(f"missing output file: {p}")

    long_units = read_jsonl(ROOT / "units" / "long_units.jsonl")
    if len(long_units) == 0:
        fail("no long units emitted")
    lu_ids = {str(r["long_unit_id"]) for r in long_units}
    if len(lu_ids) != len(long_units):
        fail("duplicate long_unit_id in long_units.jsonl")

    emb = read_jsonl(ROOT / "embeddings" / "embedding_records.jsonl")
    if len(emb) != len(long_units):
        fail(f"embedding count {len(emb)} != long unit count {len(long_units)}")
    for r in emb:
        if r["was_truncated"] is not False:
            fail(f"embedding {r['long_unit_id']} was truncated")
        if r["paper_faithful_result"] is not False:
            fail(f"embedding {r['long_unit_id']} claims paper faithfulness")
        if r["interface_validation"] is not True:
            fail(f"embedding {r['long_unit_id']} has interface_validation != true")
        dense_vector = r["dense_vector"]
        dense_dimension = r["dense_dimension"]
        if not isinstance(dense_vector, list):
            fail(f"embedding {r['long_unit_id']}: dense_vector is not a list")
        if len(dense_vector) != dense_dimension:
            fail(f"embedding {r['long_unit_id']}: dense_vector length mismatch")

    ref = read_jsonl(ROOT / "retrieval" / "reference_rankings.jsonl")
    qdr = read_jsonl(ROOT / "retrieval" / "qdrant_rankings.jsonl")
    if len(ref) == 0 or len(qdr) == 0:
        fail("retrieval outputs are empty")

    rd = read_jsonl(ROOT / "retrieval" / "ranked_documents.jsonl")
    if len(rd) == 0:
        fail("ranked documents empty")

    ec = read_jsonl(ROOT / "evaluation_cases" / "retrieval_cases.jsonl")
    if len(ec) != 6:
        fail(f"expected 6 evaluation cases, got {len(ec)}")

    for row in ec:
        qid = row.get("query_id", "<missing>")
        ranked = row.get("ranked_doc_ids")
        relevant = row.get("relevant_doc_ids")
        if not isinstance(ranked, list) or not ranked:
            fail(f"evaluation case {qid} has no ranked_doc_ids")
        if not isinstance(relevant, list) or len(relevant) != 2:
            fail(f"evaluation case {qid} must have 2 relevant docs")

    verify_checksums(ROOT / "units" / "checksums.json", ROOT / "units")
    verify_checksums(ROOT / "embeddings" / "checksums.json", ROOT / "embeddings")
    verify_checksums(ROOT / "retrieval" / "checksums.json", ROOT / "retrieval")
    verify_checksums(
        ROOT / "evaluation_cases" / "checksums.json",
        ROOT / "evaluation_cases",
    )

    print(f"[OK] {ROOT}")
    print(
        f"[OK] long_units={len(long_units)}, embeddings={len(emb)}, "
        f"ref_rankings={len(ref)}, qdrant_rankings={len(qdr)}, "
        f"ranked_docs={len(rd)}, cases={len(ec)}"
    )
    print("[OK] all checksums verified")


if __name__ == "__main__":
    main()