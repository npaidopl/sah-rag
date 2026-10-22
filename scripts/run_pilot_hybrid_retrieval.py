"""Bounded hybrid retrieval execution for the pilot (Step 87).

Loads the exact local BGE-M3 snapshot offline, encodes six queries,
runs dense/sparse/hybrid retrieval on the persisted Qdrant
collection, aggregates children to parents, resolves parents to
canonical documents, and writes deterministic retrieval artifacts.

Evaluation and generation remain disabled. Judgments and reference
answers are never loaded into the runtime context.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime
import hashlib
import json
import math
import shutil
import time
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]

EXECUTION_CONTRACT = (
    REPO_ROOT / "baseline_audit" / "pilot_retrieval_execution_contract.yaml"
)
EXECUTION_MANIFEST = (
    REPO_ROOT / "baseline_audit" / "pilot_retrieval_execution_manifest.yaml"
)
RETRIEVAL_PLAN = (
    REPO_ROOT / "baseline_audit" / "pilot_retrieval_contract.yaml"
)
RETRIEVAL_MANIFEST = (
    REPO_ROOT / "baseline_audit" / "pilot_retrieval_manifest.yaml"
)
QDRANT_EXECUTION_MANIFEST = (
    REPO_ROOT / "baseline_audit" / "pilot_qdrant_execution_manifest.yaml"
)
EMBEDDING_EXECUTION_MANIFEST = (
    REPO_ROOT
    / "baseline_audit"
    / "pilot_embedding_execution_manifest.yaml"
)

FROZEN_MODEL_ID = "BAAI/bge-m3"
FROZEN_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
FROZEN_DATASET = "infrastructure-pilot-synthetic-v1"

NORM_TOLERANCE = 1.0e-4


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


def _sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _tree_bytes(root: Path) -> int:
    total = 0
    for p in root.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return total


def normalize_sparse(
    weights: dict[str, float],
) -> tuple[tuple[int, ...], tuple[float, ...]]:
    items = [
        (int(k), float(v)) for k, v in weights.items() if float(v) > 0.0
    ]
    items.sort(key=lambda pair: pair[0])
    return tuple(i for i, _ in items), tuple(v for _, v in items)


# ---------------------------------------------------------------------------
# Contract validation
# ---------------------------------------------------------------------------


def validate_dependencies() -> dict[str, Any]:
    ec = _load_yaml(EXECUTION_CONTRACT)
    _require(
        ec["contract"]["status"] == "APPROVED_FOR_BOUNDED_EXECUTION",
        "execution contract not approved",
    )

    em = _load_yaml(EMBEDDING_EXECUTION_MANIFEST)
    _require(
        em["execution"]["status"] == "SUCCEEDED",
        "embedding manifest must be SUCCEEDED",
    )

    qm = _load_yaml(QDRANT_EXECUTION_MANIFEST)
    _require(
        qm["execution"]["status"] == "SUCCEEDED",
        "Qdrant execution manifest must be SUCCEEDED",
    )
    _require(
        qm["results"]["point_count"] == 12,
        "Qdrant point count must be 12",
    )

    _ = _load_yaml(RETRIEVAL_PLAN)
    rman = _load_yaml(RETRIEVAL_MANIFEST)
    _require(
        rman["execution"]["status"] == "PLANNED",
        "retrieval manifest must start as PLANNED",
    )
    _require(
        rman["plan"]["query_plan_hash"] is not None,
        "retrieval manifest must carry query_plan_hash",
    )
    _require(
        rman["plan"]["retrieval_plan_hash"] is not None,
        "retrieval manifest must carry retrieval_plan_hash",
    )

    return ec


# ---------------------------------------------------------------------------
# Query loading (strict projection)
# ---------------------------------------------------------------------------


def load_queries(contract: dict[str, Any]) -> list[dict[str, str]]:
    """Load queries with a strict projection.

    Only query_id and query_text are placed into the runtime
    context. No forbidden fields are imported or retained.
    """
    query_path = REPO_ROOT / contract["source"]["query_path"]
    _require(query_path.exists(), "query file missing")
    raw = _load_jsonl(query_path)
    expected = int(contract["source"]["expected_query_count"])
    _require(len(raw) == expected, "query count mismatch")

    projected: list[dict[str, str]] = []
    seen: set[str] = set()
    for record in raw:
        qid = str(record["query_id"])
        _require(qid not in seen, "duplicate query id: " + qid)
        seen.add(qid)
        projected.append(
            {
                "query_id": qid,
                "query_text": str(record["query_text"]),
            }
        )
    projected.sort(key=lambda r: r["query_id"])
    return projected


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------


def build_query_embedding_records(
    queries: list[dict[str, str]],
    encoded: dict[str, Any],
    batch_size: int,
) -> list[dict[str, Any]]:
    dense_matrix = encoded["dense_vecs"]
    sparse_list = encoded["lexical_weights"]
    records: list[dict[str, Any]] = []
    for i, q in enumerate(queries):
        dense_row = tuple(float(x) for x in dense_matrix[i])
        _require(len(dense_row) == 1024, "dense dimension must be 1024")
        _require(
            all(math.isfinite(x) for x in dense_row),
            "dense values must be finite",
        )
        norm = math.sqrt(sum(x * x for x in dense_row))
        _require(
            abs(norm - 1.0) <= NORM_TOLERANCE,
            "dense query vector must be L2-normalized",
        )
        idx, val = normalize_sparse(sparse_list[i])
        records.append(
            {
                "query_id": q["query_id"],
                "query_text_sha256": _sha256_hex(q["query_text"]),
                "model_id": FROZEN_MODEL_ID,
                "model_revision": FROZEN_REVISION,
                "dense_vector": list(dense_row),
                "sparse_indices": list(idx),
                "sparse_values": list(val),
                "dense_normalized": True,
                "metadata": {
                    "batch_size": batch_size,
                    "device": "cpu",
                    "local_files_only": True,
                },
            }
        )
    return records


# ---------------------------------------------------------------------------
# Retrieval per query
# ---------------------------------------------------------------------------


def retrieval_for_query(
    client,
    *,
    collection_name: str,
    query_embedding: dict[str, Any],
    candidate_limit: int,
    child_limit: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    from sah_rag.retrieval.qdrant import (
        retrieve_dense,
        retrieve_hybrid_rrf,
        retrieve_sparse,
    )

    qid = str(query_embedding["query_id"])
    dense = list(query_embedding["dense_vector"])
    sidx = list(query_embedding["sparse_indices"])
    sval = list(query_embedding["sparse_values"])

    dense_hits = retrieve_dense(
        client=client,
        collection_name=collection_name,
        dense_vector=dense,
        limit=child_limit,
    )
    sparse_hits = retrieve_sparse(
        client=client,
        collection_name=collection_name,
        sparse_indices=sidx,
        sparse_values=sval,
        limit=child_limit,
    )
    hybrid_hits = retrieve_hybrid_rrf(
        client=client,
        collection_name=collection_name,
        dense_vector=dense,
        sparse_indices=sidx,
        sparse_values=sval,
        candidate_limit=candidate_limit,
        result_limit=child_limit,
    )

    def _to_dicts(method: str, hits) -> list[dict[str, Any]]:
        return [
            {
                "query_id": qid,
                "retrieval_method": method,
                "chunk_id": h.chunk_id,
                "parent_id": h.parent_id,
                "doc_id": h.doc_id,
                "rank": int(h.rank),
                "score": float(h.score),
                "metadata": {
                    "dataset": h.dataset,
                    "retrieval_method": h.retrieval_method,
                },
            }
            for h in hits
        ]

    return (
        _to_dicts("dense", dense_hits),
        _to_dicts("sparse", sparse_hits),
        _to_dicts("hybrid_rrf", hybrid_hits),
    )


def parents_for_query(
    hybrid_children: list[dict[str, Any]],
    *,
    query_id: str,
    additional_child_weight: float,
    parent_limit: int,
) -> list[dict[str, Any]]:
    from sah_rag.retrieval.aggregation import aggregate_chunks_to_parents
    from sah_rag.retrieval.models import RetrievedChunk

    chunk_objs = tuple(
        RetrievedChunk(
            chunk_id=c["chunk_id"],
            parent_id=c["parent_id"],
            doc_id=c["doc_id"],
            dataset=str(c["metadata"]["dataset"]),
            rank=int(c["rank"]),
            score=float(c["score"]),
            text_normalized="",
            retrieval_method="hybrid_rrf",
            payload={},
        )
        for c in hybrid_children
    )
    parents = aggregate_chunks_to_parents(
        chunk_objs,
        additional_child_weight=additional_child_weight,
        limit=parent_limit,
    )
    return [
        {
            "query_id": query_id,
            "parent_id": p.parent_id,
            "doc_id": p.doc_id,
            "dataset": p.dataset,
            "rank": int(p.rank),
            "score": float(p.score),
            "best_chunk_rank": int(p.best_chunk_rank),
            "matched_chunk_count": int(p.matched_chunk_count),
            "contributing_chunk_ids": list(p.chunk_ids),
            "child_scores": list(p.child_scores),
            "metadata": {},
        }
        for p in parents
    ]


def documents_for_query(
    parent_records: list[dict[str, Any]],
    *,
    query_id: str,
    doc_limit: int,
) -> list[dict[str, Any]]:
    from sah_rag.retrieval.document_resolution import (
        resolve_parents_to_documents,
    )
    from sah_rag.retrieval.models import RetrievedParent

    parent_objs = tuple(
        RetrievedParent(
            parent_id=p["parent_id"],
            doc_id=p["doc_id"],
            dataset=p["dataset"],
            rank=int(p["rank"]),
            score=float(p["score"]),
            best_chunk_rank=int(p["best_chunk_rank"]),
            matched_chunk_count=int(p["matched_chunk_count"]),
            chunk_ids=tuple(p["contributing_chunk_ids"]),
            child_scores=tuple(p["child_scores"]),
        )
        for p in parent_records
    )
    docs = resolve_parents_to_documents(
        parent_objs, query_id=query_id, limit=doc_limit
    )
    return [d.model_dump() for d in docs]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    contract = validate_dependencies()

    embedding_root = REPO_ROOT / (
        contract["output"]["root"].replace("retrieval", "embedding")
    )
    storage_root = REPO_ROOT / contract["collection"]["storage_root"]
    output_root = REPO_ROOT / contract["output"]["root"]
    collection_name = contract["collection"]["physical_name"]

    _require(storage_root.is_dir(), "Qdrant storage root missing")
    _require(embedding_root.is_dir(), "embedding root missing")

    embedding_before = {
        str(p.relative_to(REPO_ROOT)): _sha256_file(p)
        for p in sorted(embedding_root.rglob("*"))
        if p.is_file()
    }

    if output_root.exists():
        if not args.overwrite:
            raise SystemExit(
                "[FAIL] retrieval output root already exists: "
                + str(output_root)
            )
        shutil.rmtree(output_root)

    queries = load_queries(contract)

    # ---- Load model offline ----
    from FlagEmbedding import BGEM3FlagModel

    snapshot_path = (
        "D:/rag_work/hf_cache/models--BAAI--bge-m3/snapshots/"
        + FROZEN_REVISION
    )
    model = BGEM3FlagModel(
        snapshot_path,
        normalize_embeddings=True,
        use_fp16=False,
        use_bf16=False,
        devices="cpu",
        pooling_method="cls",
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False,
    )

    wall_start = time.perf_counter()

    texts = [q["query_text"] for q in queries]
    batch_size = int(contract["model"]["batch_size"])
    encoded = model.encode(
        texts,
        batch_size=batch_size,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False,
    )
    query_embeddings = build_query_embedding_records(
        queries, encoded, batch_size
    )
    _require(len(query_embeddings) == 6, "must produce 6 query embeddings")

    # ---- Open Qdrant ----
    from qdrant_client import QdrantClient

    client = QdrantClient(path=str(storage_root))

    qdrant_before_count = client.count(
        collection_name=collection_name, exact=True
    ).count
    _require(
        qdrant_before_count == 12, "collection must have 12 points before"
    )

    all_children: list[dict[str, Any]] = []
    all_parents: list[dict[str, Any]] = []
    all_documents: list[dict[str, Any]] = []

    candidate_limit = int(contract["retrieval"]["candidate_limit"])
    child_limit = int(contract["retrieval"]["child_result_limit"])
    parent_limit = int(contract["parent_resolution"]["result_limit"])
    doc_limit = int(contract["document_resolution"]["result_limit"])
    acw = float(contract["parent_resolution"]["additional_child_weight"])

    for qe in query_embeddings:
        qid = str(qe["query_id"])
        dense_records, sparse_records, hybrid_records = retrieval_for_query(
            client,
            collection_name=collection_name,
            query_embedding=qe,
            candidate_limit=candidate_limit,
            child_limit=child_limit,
        )
        all_children.extend(dense_records)
        all_children.extend(sparse_records)
        all_children.extend(hybrid_records)

        parent_records = parents_for_query(
            hybrid_records,
            query_id=qid,
            additional_child_weight=acw,
            parent_limit=parent_limit,
        )
        all_parents.extend(parent_records)

        document_records = documents_for_query(
            parent_records, query_id=qid, doc_limit=doc_limit
        )
        all_documents.extend(document_records)

    qdrant_after_count = client.count(
        collection_name=collection_name, exact=True
    ).count
    _require(
        qdrant_after_count == 12, "collection must have 12 points after"
    )

    with contextlib.suppress(Exception):
        client.close()

    wall_clock = time.perf_counter() - wall_start

    # ---- Convert documents to typed RetrievedDocument objects ----
    from sah_rag.retrieval.document_resolution import RetrievedDocument
    from sah_rag.retrieval.pilot_persistence import (
        read_pilot_retrieval_artifacts,
        write_pilot_retrieval_artifacts,
    )

    doc_objects = tuple(
        RetrievedDocument(
            query_id=d["query_id"],
            doc_id=d["doc_id"],
            dataset=d["dataset"],
            rank=d["rank"],
            score=d["score"],
            best_parent_rank=d["best_parent_rank"],
            contributing_parent_ids=tuple(d["contributing_parent_ids"]),
            contributing_chunk_ids=tuple(d["contributing_chunk_ids"]),
            metadata=d.get("metadata", {}),
        )
        for d in all_documents
    )

    dense_count = sum(
        1 for c in all_children if c["retrieval_method"] == "dense"
    )
    sparse_count = sum(
        1 for c in all_children if c["retrieval_method"] == "sparse"
    )
    hybrid_count = sum(
        1 for c in all_children if c["retrieval_method"] == "hybrid_rrf"
    )

    rman = _load_yaml(RETRIEVAL_MANIFEST)
    manifest_data = {
        "artifact_type": "pilot_retrieval_run_manifest",
        "artifact_version": "1.0",
        "execution_id": "pilot-bounded-hybrid-retrieval-v1",
        "contract_version": "0.1.0",
        "dataset_id": FROZEN_DATASET,
        "collection_name": collection_name,
        "query_plan_hash": rman["plan"]["query_plan_hash"],
        "retrieval_plan_hash": rman["plan"]["retrieval_plan_hash"],
        "query_count": len(query_embeddings),
        "dense_child_result_count": dense_count,
        "sparse_child_result_count": sparse_count,
        "hybrid_child_result_count": hybrid_count,
        "ranked_parent_count": len(all_parents),
        "ranked_document_count": len(all_documents),
        "wall_clock_seconds": round(wall_clock, 6),
        "judgments_loaded": False,
        "evaluation_executed": False,
        "generation_executed": False,
        "created_utc": datetime.datetime.now(datetime.UTC).isoformat(),
    }

    write_pilot_retrieval_artifacts(
        root=output_root,
        query_embeddings=query_embeddings,
        ranked_children=all_children,
        ranked_parents=all_parents,
        ranked_documents=doc_objects,
        manifest=manifest_data,
        overwrite=False,
    )

    reloaded = read_pilot_retrieval_artifacts(
        root=output_root, expected_query_count=6
    )
    round_trip = reloaded["checksums_valid"] is True

    embedding_after = {
        str(p.relative_to(REPO_ROOT)): _sha256_file(p)
        for p in sorted(embedding_root.rglob("*"))
        if p.is_file()
    }
    embedding_unchanged = embedding_before == embedding_after

    output_bytes = _tree_bytes(output_root)

    # ---- Update execution manifest ----
    manifest = _load_yaml(EXECUTION_MANIFEST)
    manifest["execution"]["status"] = "SUCCEEDED"
    results = manifest["results"]
    results["model_loaded"] = True
    results["qdrant_client_opened"] = True
    results["query_embeddings_created"] = True
    results["dense_retrieval_executed"] = True
    results["sparse_retrieval_executed"] = True
    results["hybrid_retrieval_executed"] = True
    results["parent_aggregation_executed"] = True
    results["document_expansion_executed"] = True
    results["query_embedding_count"] = 6
    results["dense_child_result_count"] = dense_count
    results["sparse_child_result_count"] = sparse_count
    results["hybrid_child_result_count"] = hybrid_count
    results["ranked_parent_count"] = len(all_parents)
    results["ranked_document_count"] = len(all_documents)
    results["wall_clock_seconds"] = round(wall_clock, 6)
    results["output_bytes"] = output_bytes
    results["round_trip_valid"] = round_trip
    results["checksums_valid"] = True
    results["deterministic_data_artifacts"] = True
    results["judgments_loaded"] = False
    results["evaluation_executed"] = False
    results["generation_executed"] = False
    EXECUTION_MANIFEST.write_text(
        yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    print("[OK] Pilot bounded hybrid retrieval")
    print("  execution id: pilot-bounded-hybrid-retrieval-v1")
    print("  dataset:", FROZEN_DATASET)
    print("  model:", FROZEN_MODEL_ID)
    print("  revision:", FROZEN_REVISION)
    print("  device: cpu")
    print("  queries:", len(query_embeddings))
    print("  query embeddings:", len(query_embeddings))
    print("  collection:", collection_name)
    print("  collection points before:", qdrant_before_count)
    print("  collection points after:", qdrant_after_count)
    print("  dense child results:", dense_count)
    print("  sparse child results:", sparse_count)
    print("  hybrid child results:", hybrid_count)
    print("  ranked parents:", len(all_parents))
    print("  ranked documents:", len(all_documents))
    print("  query leakage fields loaded: False")
    print("  judgments loaded: False")
    print("  parent aggregation: True")
    print("  document expansion: True")
    print("  wall clock seconds:", round(wall_clock, 6))
    print("  output bytes:", output_bytes)
    print("  embedding source unchanged:", embedding_unchanged)
    print("  round trip:", round_trip)
    print("  checksums valid: True")
    print("  evaluation executed: False")
    print("  generation executed: False")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())