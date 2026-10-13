"""Validate baseline_audit/longrag_controlled_embedding_contract.yaml."""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

PATH = Path("baseline_audit/longrag_controlled_embedding_contract.yaml")

REQUIRED_TOP_KEYS = {
    "artifact_type", "artifact_version", "contract", "architecture",
    "controlled_component", "model", "execution", "fidelity",
    "acceptance", "safety", "claims",
}


def fail(msg: str) -> None:
    print(f"[FAIL] {msg}")
    sys.exit(1)


def main() -> None:
    if not PATH.exists():
        fail(f"missing: {PATH}")

    d = yaml.safe_load(PATH.read_text(encoding="utf-8"))

    missing = REQUIRED_TOP_KEYS - set(d.keys())
    if missing:
        fail(f"missing top-level keys: {sorted(missing)}")

    if d["artifact_type"] != "longrag_controlled_embedding_contract":
        fail("wrong artifact_type")
    if d["artifact_version"] != "1.0":
        fail("wrong artifact_version")

    if d["contract"]["status"] != "APPROVED_FOR_BOUNDED_EXECUTION":
        fail("contract.status must be APPROVED_FOR_BOUNDED_EXECUTION")

    arch = d["architecture"]
    if arch["architecture_id"] != "longrag":
        fail("architecture_id must be longrag")
    if arch["fidelity_class"] != "CONTROLLED_ADAPTATION":
        fail("fidelity_class must be CONTROLLED_ADAPTATION")

    model = d["model"]
    if model["model_id"] != "BAAI/bge-m3":
        fail("model_id must be BAAI/bge-m3")
    if model["model_revision"] != "5617a9f61b028005a4858fdac845db406aefb181":
        fail("model_revision must be pinned")
    if model["local_snapshot_required"] is not True:
        fail("local_snapshot_required must be true")
    if model["network_fallback_allowed"] is not False:
        fail("network_fallback_allowed must be false")

    exec_ = d["execution"]
    if exec_["dense_enabled"] is not True:
        fail("dense_enabled must be true")
    if exec_["sparse_enabled"] is not False:
        fail("sparse_enabled must be false")
    if exec_["multi_vector_enabled"] is not False:
        fail("multi_vector_enabled must be false")
    if exec_["dense_dimension"] != 1024:
        fail("dense_dimension must be 1024")
    if exec_["dense_normalized"] is not True:
        fail("dense_normalized must be true")
    if exec_["use_fp16"] is not False:
        fail("use_fp16 must be false")
    if exec_["use_bf16"] is not False:
        fail("use_bf16 must be false")

    fid = d["fidelity"]
    if fid["native_embedder_identity"] != "UNPINNED":
        fail("native_embedder_identity must be UNPINNED")
    if fid["native_embedder_executed"] is not False:
        fail("native_embedder_executed must be false")
    if fid["native_equivalence_claimed"] is not False:
        fail("native_equivalence_claimed must be false")

    claims = d["claims"]
    if claims["bge_m3_is_original_longrag_embedder"] is not False:
        fail("must not claim bge-m3 is the original longrag embedder")
    if claims["bge_m3_is_proven_superior_for_longrag"] is not False:
        fail("must not claim bge-m3 is proven superior")
    if claims["architecture_behavior_under_controlled_encoder"] is not True:
        fail("must claim architecture_behavior_under_controlled_encoder")

    safety = d["safety"]
    for k in ("model_download_enabled", "network_access_enabled", "hosted_requests_enabled"):
        if safety[k] is not False:
            fail(f"safety.{k} must be false")

    print(f"[OK] {PATH}")
    print(f"[OK] status=APPROVED, model=bge-m3@{model['model_revision'][:12]}, dense-only")


if __name__ == "__main__":
    main()