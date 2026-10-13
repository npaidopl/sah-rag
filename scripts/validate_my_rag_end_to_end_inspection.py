"""Validate the rag_system end-to-end static inspection."""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

AUDIT = Path("baseline_audit/my_rag_end_to_end_inspection.yaml")

REQUIRED_OFFLINE_STAGES = {
    "canonical_input_validation",
    "derived_preprocessing",
    "structural_parsing",
    "semantic_parent_construction",
    "hierarchical_child_chunking",
    "dense_and_sparse_embedding",
    "vector_store_ingestion",
    "parent_and_document_storage",
}

REQUIRED_ONLINE_STAGES = {
    "canonical_query_loading",
    "dense_query_embedding",
    "sparse_query_embedding",
    "dense_retrieval",
    "sparse_retrieval",
    "rank_fusion",
    "top_k_child_retrieval",
    "semantic_parent_resolution",
    "canonical_document_resolution",
    "evaluation",
}

ALLOWED_EXECUTABILITY = {"true", "partial", "false"}


def main() -> int:
    if not AUDIT.is_file():
        raise SystemExit(f"missing audit: {AUDIT}")
    data = yaml.safe_load(AUDIT.read_text(encoding="utf-8"))

    pipeline = data["pipeline"]
    if pipeline["ownership"] != "PROJECT_OWNED":
        raise SystemExit("ownership must be PROJECT_OWNED")
    if pipeline["verification_level"] != "STATIC_CODE_INSPECTION":
        raise SystemExit("verification_level must be STATIC_CODE_INSPECTION")
    if not pipeline["root"]:
        raise SystemExit("root must be non-empty")

    offline = data["stages"]["offline"]
    online = data["stages"]["online"]
    offline_ids = {s["stage_id"] for s in offline}
    online_ids = {s["stage_id"] for s in online}
    if offline_ids != REQUIRED_OFFLINE_STAGES:
        raise SystemExit("offline stage set changed")
    if online_ids != REQUIRED_ONLINE_STAGES:
        raise SystemExit("online stage set changed")

    for s in offline + online:
        exe = s["executable"]
        if isinstance(exe, bool):
            exe = "true" if exe else "false"
        if str(exe) not in ALLOWED_EXECUTABILITY:
            raise SystemExit(f"{s['stage_id']}: invalid executable value")

    ck = data["chunking"]
    if ck["chunk_size_tokens"] != 256:
        raise SystemExit("chunk_size_tokens must be 256")
    if ck["chunk_overlap_tokens"] != 32:
        raise SystemExit("chunk_overlap_tokens must be 32")
    if ck["deterministic"] is not True:
        raise SystemExit("chunking.deterministic must be true")
    if ck["parent_boundaries_preserved"] is not True:
        raise SystemExit("chunking.parent_boundaries_preserved must be true")

    emb = data["embedding"]
    if emb["dense_model_id"] != "BAAI/bge-m3":
        raise SystemExit("dense_model_id must be BAAI/bge-m3")
    if emb["dense_dimension"] != 1024:
        raise SystemExit("dense_dimension must be 1024")
    if emb["local_or_hosted"] != "local":
        raise SystemExit("local_or_hosted must be local")
    if emb["reproducibility_status"] not in ("PINNED", "MIXED", "UNPINNED"):
        raise SystemExit("reproducibility_status invalid")

    st = data["storage"]
    if st["backend"] != "Qdrant":
        raise SystemExit("storage.backend must be Qdrant")
    if st["unsafe_serialization"] is not False:
        raise SystemExit("unsafe_serialization must be false")

    rb = data["retrieval"]
    if rb["score_direction"] != "higher_is_better":
        raise SystemExit("score_direction must be higher_is_better")
    if rb["canonical_doc_ids_exposed"] is not True:
        raise SystemExit("canonical_doc_ids_exposed must be true")

    gb = data["generation_boundary"]
    if gb["retrieval_callable_independently"] is not True:
        raise SystemExit("retrieval_callable_independently must be true")
    if gb["final_generation_enabled_for_pilot"] is not False:
        raise SystemExit("final_generation_enabled_for_pilot must be false")

    cc = data["canonical_compatibility"]
    if cc["status"] not in ("DIRECT", "ADAPTER_REQUIRED", "PARTIAL", "INCOMPATIBLE"):
        raise SystemExit("canonical_compatibility.status invalid")
    if cc["adapter_required"] and not cc.get("adapter_scope"):
        raise SystemExit("adapter_scope must be non-empty when adapter_required")

    hf = data["host_feasibility"]
    if hf["current_host_status"] not in (
        "EXECUTABLE_ON_CURRENT_HOST",
        "EXECUTABLE_ON_CURRENT_HOST_WITH_LIMITS",
        "REQUIRES_REMOTE_GPU",
        "NOT_EXECUTABLE",
    ):
        raise SystemExit("host_feasibility status invalid")

    pr = data["pilot_recommendation"]
    if not isinstance(pr["eligible"], bool):
        raise SystemExit("pilot_recommendation.eligible must be boolean")
    if pr["eligible"] and not pr.get("required_changes"):
        raise SystemExit("required_changes must be non-empty when eligible")

    if len(data["blocked_claims"]) < 4:
        raise SystemExit("blocked_claims must have at least four entries")

    print("[OK] rag_system end-to-end static inspection")
    print(f"  root: {pipeline['root']}")
    print(f"  source files: {data['source_counts']['total_files']}")
    print(f"  python files: {data['source_counts']['python_files']}")
    print(f"  offline stages: {len(offline)}")
    print(f"  online stages: {len(online)}")
    print(f"  embedding provider: {emb['provider']}")
    print(f"  embedding model: {emb['dense_model_id']}")
    print(f"  storage backend: {st['backend']}")
    print(f"  chunking: {ck['chunk_size_tokens']} / {ck['chunk_overlap_tokens']}")
    print(f"  retrieval fusion: {rb['rank_fusion_method']}")
    print(f"  canonical compatibility: {cc['status']}")
    print(f"  adapter required: {cc['adapter_required']}")
    print(f"  CPU-only feasibility: {hf['cpu_only_supported']}")
    print(f"  pilot eligibility: {pr['eligible']}")
    print(f"  generation enabled for pilot: {gb['final_generation_enabled_for_pilot']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
