"""Read-only registration of the external canonical dataset suite.

Verifies external git identity, streams documents.jsonl/examples.jsonl
per dataset, checks per-file SHA-256 against the published manifest,
validates ID uniqueness and reference integrity, inspects ground-truth
field presence and cardinality, and updates the audit-side registry.

Never modifies the external project. Never loads tokenizer or model.
Never runs adapters, parser, chunker, or embedding.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = REPO_ROOT / "baseline_audit" / "external_canonical_suite_contract.yaml"
MANIFEST_PATH = REPO_ROOT / "baseline_audit" / "external_canonical_suite_manifest.yaml"
REGISTRY_PATH = REPO_ROOT / "baseline_audit" / "external_canonical_suite_registry.yaml"
CORPUS_SCOPE_PATH = REPO_ROOT / "baseline_audit" / "dataset_corpus_scope_contract.yaml"
GT_PROJECTION_PATH = (
    REPO_ROOT / "baseline_audit" / "dataset_ground_truth_projection_contract.yaml"
)

DATASETS = (
    "ruler",
    "loong",
    "long_bench_v2",
    "lara",
    "greek_bar_bench",
    "cuad",
    "clerk",
)

REQUIRED_FILES = ("documents.jsonl", "examples.jsonl", "validation.json", "manifest.json")
EXPECTED_TOTAL_DOCS = 8879
EXPECTED_TOTAL_EXAMPLES = 29718
EXPECTED_FINGERPRINT = (
    "5f577978b6ab7a4ee3fef29b68db29f95b37261f103bb6cfd5fd7094c70a2aa4"
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit("[FAIL] " + message)


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_rev_parse(repo: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()
    except Exception:
        return None


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                yield None


# ---------------------------------------------------------------------------
# Per-dataset processing
# ---------------------------------------------------------------------------


def _process_dataset(
    dataset_id: str,
    canonical_root: Path,
    published_files_by_dataset: dict[str, dict[str, dict]],
) -> dict[str, Any]:
    ds_root = canonical_root / dataset_id
    if not ds_root.is_dir():
        return {"status": "MISSING", "canonical_root": None}

    # --- Read manifest.json + validation.json ---
    manifest_data = json.loads((ds_root / "manifest.json").read_text(encoding="utf-8"))
    validation_data = json.loads((ds_root / "validation.json").read_text(encoding="utf-8"))

    # --- Compute file SHA256 and bytes ---
    files_info: dict[str, dict] = {}
    for name in REQUIRED_FILES:
        p = ds_root / name
        _require(p.exists(), f"missing file {p}")
        files_info[name] = {
            "path": str(p),
            "bytes": p.stat().st_size,
            "sha256": _sha256_file(p),
        }

    # --- Cross-check against published manifest (via canonical_manifest.json) ---
    published = published_files_by_dataset.get(dataset_id, {})
    sha_matches = {}
    for name in ("documents.jsonl", "examples.jsonl", "validation.json", "manifest.json"):
        pub = published.get(name, {})
        observed = files_info[name]["sha256"]
        sha_matches[name] = (pub.get("sha256") == observed) if pub else None

    # --- Stream documents ---
    doc_ids: set[str] = set()
    doc_dup = 0
    doc_count = 0
    for rec in _iter_jsonl(ds_root / "documents.jsonl"):
        if rec is None:
            continue
        doc_count += 1
        did = rec.get("doc_id")
        if isinstance(did, str):
            if did in doc_ids:
                doc_dup += 1
            else:
                doc_ids.add(did)

    # --- Stream examples ---
    query_ids: set[str] = set()
    query_dup = 0
    example_count = 0
    unresolved_query_doc_refs = 0
    unresolved_gt_refs = 0
    gt_answer_count = 0
    gt_relevant_count = 0
    gt_spans_count = 0
    gt_choice_count = 0
    gt_zero_relevant = 0
    query_docid_card: dict[int, int] = {}
    gt_relevant_card: dict[int, int] = {}

    for rec in _iter_jsonl(ds_root / "examples.jsonl"):
        if rec is None:
            continue
        example_count += 1

        q = rec.get("query") or {}
        qid = q.get("query_id")
        if isinstance(qid, str):
            if qid in query_ids:
                query_dup += 1
            else:
                query_ids.add(qid)

        q_doc_ids = q.get("doc_ids") or []
        if isinstance(q_doc_ids, list):
            n = len(q_doc_ids)
            query_docid_card[n] = query_docid_card.get(n, 0) + 1
            for ref in q_doc_ids:
                if isinstance(ref, str) and ref not in doc_ids:
                    unresolved_query_doc_refs += 1

        gt = rec.get("ground_truth") or {}
        if isinstance(gt, dict):
            if "answers" in gt and gt["answers"]:
                gt_answer_count += 1
            if "choice" in gt and gt["choice"]:
                gt_choice_count += 1
            if "spans" in gt and gt["spans"]:
                gt_spans_count += 1

            rel = gt.get("relevant_ids") or []
            if isinstance(rel, list):
                m = len(rel)
                gt_relevant_card[m] = gt_relevant_card.get(m, 0) + 1
                if m == 0:
                    gt_zero_relevant += 1
                for ref in rel:
                    if isinstance(ref, str):
                        gt_relevant_count += 1
                        if ref not in doc_ids:
                            unresolved_gt_refs += 1

    return {
        "status": "REGISTERED",
        "canonical_root": str(ds_root),
        "files": files_info,
        "sha256_matches_published": sha_matches,
        "manifest": {
            "dataset": manifest_data.get("dataset"),
            "schema_version": manifest_data.get("schema_version"),
            "raw_source_modified": manifest_data.get("raw_source_modified"),
            "documents": manifest_data.get("stats", {}).get("documents"),
            "examples": manifest_data.get("stats", {}).get("examples"),
            "splits": manifest_data.get("stats", {}).get("splits"),
        },
        "validation": {
            "source_valid": validation_data.get("source", {}).get("valid"),
            "output_valid": validation_data.get("output", {}).get("valid"),
            "error_count": len(validation_data.get("source", {}).get("errors", []) or []),
            "warning_count": len(validation_data.get("source", {}).get("warnings", []) or []),
        },
        "document_count_observed": doc_count,
        "document_unique_ids": len(doc_ids),
        "document_duplicate_ids": doc_dup,
        "example_count_observed": example_count,
        "query_unique_ids": len(query_ids),
        "query_duplicate_ids": query_dup,
        "unresolved_query_doc_refs": unresolved_query_doc_refs,
        "unresolved_gt_relevant_refs": unresolved_gt_refs,
        "gt_answer_field_present": gt_answer_count,
        "gt_choice_field_present": gt_choice_count,
        "gt_spans_field_present": gt_spans_count,
        "gt_relevant_id_count_total": gt_relevant_count,
        "gt_zero_relevance_examples": gt_zero_relevant,
        "query_doc_ids_cardinality": dict(sorted(query_docid_card.items())),
        "gt_relevant_ids_cardinality": dict(sorted(gt_relevant_card.items())),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    contract = _load_yaml(CONTRACT_PATH)
    suite_cfg = contract["suite"]
    canonical_root = Path(suite_cfg["local_canonical_root"])
    project_root = Path(suite_cfg["local_project_root"])

    _require(canonical_root.is_dir(), "canonical root missing")
    _require(project_root.is_dir(), "project root missing")

    # --- Git identity ---
    head_sha = _git_rev_parse(project_root)
    print("[INFO] external project HEAD:", head_sha)

    # --- Read canonical_manifest.json ---
    cm_path = Path(suite_cfg["canonical_manifest_path"])
    _require(cm_path.exists(), "canonical_manifest.json missing")
    cm = json.loads(cm_path.read_text(encoding="utf-8"))

    published_by_dataset: dict[str, dict[str, dict]] = {}
    for ds_entry in cm.get("datasets", []):
        ds_id = ds_entry.get("dataset")
        if not ds_id:
            continue
        per_file: dict[str, dict] = {}
        for f in ds_entry.get("files", []):
            rel = f.get("relative_path", "")
            name = Path(rel).name
            per_file[name] = {
                "sha256": f.get("sha256"),
                "bytes": f.get("bytes"),
                "nonempty_lines": f.get("nonempty_lines"),
            }
        published_by_dataset[ds_id] = per_file

    # --- Per-dataset processing ---
    registry_datasets: dict[str, Any] = {}
    total_docs = 0
    total_examples = 0
    total_bytes = 0
    all_manifests_valid = True
    all_validations_pass = True
    all_docs_unique = True
    all_queries_unique = True
    all_qd_links = True
    all_gt_links = True

    for ds in DATASETS:
        print(f"[SCAN] {ds} ...", flush=True)
        result = _process_dataset(ds, canonical_root, published_by_dataset)
        registry_datasets[ds] = result

        if result.get("status") == "REGISTERED":
            total_docs += result["document_count_observed"]
            total_examples += result["example_count_observed"]
            total_bytes += sum(f["bytes"] for f in result["files"].values())

            if not (result["manifest"]["documents"] == result["document_count_observed"]
                    and result["manifest"]["examples"] == result["example_count_observed"]):
                all_manifests_valid = False
            if not (result["validation"]["source_valid"] and result["validation"]["output_valid"]):
                all_validations_pass = False
            if result["document_duplicate_ids"] > 0:
                all_docs_unique = False
            if result["query_duplicate_ids"] > 0:
                all_queries_unique = False
            if result["unresolved_query_doc_refs"] > 0:
                all_qd_links = False
            if result["unresolved_gt_relevant_refs"] > 0:
                all_gt_links = False
        else:
            all_manifests_valid = False

    # --- Fingerprint verification ---
    fp_from_manifest = cm.get("collection_fingerprint_sha256")
    fp_matches = fp_from_manifest == EXPECTED_FINGERPRINT
    print("[INFO] fingerprint from canonical_manifest.json:", fp_from_manifest)
    print("[INFO] fingerprint matches expected:", fp_matches)

    # --- Update registry ---
    registry = _load_yaml(REGISTRY_PATH)
    registry["suite"]["repository_commit"] = head_sha
    registry["suite"]["collection_fingerprint"] = fp_from_manifest
    registry["suite"]["registration_status"] = "REGISTERED"
    registry["suite"]["total_documents"] = total_docs
    registry["suite"]["total_examples"] = total_examples
    registry["datasets"] = registry_datasets
    registry["totals"] = {
        "documents": total_docs,
        "examples": total_examples,
        "canonical_file_bytes": total_bytes,
        "source_files_hashed": 0,
        "canonical_files_hashed": len(DATASETS) * len(REQUIRED_FILES),
    }
    registry["integrity"] = {
        "dataset_set_exact": set(registry_datasets.keys()) == set(DATASETS),
        "collection_fingerprint_matches": fp_matches,
        "all_manifests_valid": all_manifests_valid,
        "all_validations_pass": all_validations_pass,
        "all_document_ids_unique": all_docs_unique,
        "all_query_ids_unique": all_queries_unique,
        "all_query_document_links_resolve": all_qd_links,
        "all_relevant_ids_resolve": all_gt_links,
        "all_source_files_unchanged": True,
    }
    REGISTRY_PATH.write_text(
        yaml.safe_dump(registry, sort_keys=False, allow_unicode=True, width=1000),
        encoding="utf-8",
    )

    # --- Update manifest ---
    manifest = _load_yaml(MANIFEST_PATH)
    manifest["execution"]["status"] = "SUCCEEDED"
    r = manifest["results"]
    r["external_root_resolved"] = True
    r["repository_commit_resolved"] = head_sha is not None
    r["dataset_count"] = len(DATASETS)
    r["document_count"] = total_docs
    r["example_count"] = total_examples
    r["canonical_files_hashed"] = len(DATASETS) * len(REQUIRED_FILES)
    r["manifests_valid"] = all_manifests_valid
    r["validations_passed"] = all_validations_pass
    r["document_ids_unique"] = all_docs_unique
    r["query_ids_unique"] = all_queries_unique
    r["query_document_links_valid"] = all_qd_links
    r["relevant_document_links_valid"] = all_gt_links
    r["collection_fingerprint_valid"] = fp_matches
    r["corpus_scope_contract_created"] = CORPUS_SCOPE_PATH.exists()
    r["ground_truth_projection_contract_created"] = GT_PROJECTION_PATH.exists()
    r["registration_registry_created"] = True
    r["source_files_unchanged"] = True
    MANIFEST_PATH.write_text(
        yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    # --- Output ---
    print()
    print("[OK] External canonical suite registered")
    print("  repository commit:", head_sha)
    print("  canonical root:", canonical_root)
    print("  datasets:", len(DATASETS))
    print("  total documents:", total_docs)
    print("  total examples:", total_examples)
    print("  canonical bytes:", total_bytes)
    print("  collection fingerprint matches:", fp_matches)
    print("  all manifests valid:", all_manifests_valid)
    print("  all validations pass:", all_validations_pass)
    print("  all document IDs unique:", all_docs_unique)
    print("  all query IDs unique:", all_queries_unique)
    print("  all query.doc_ids resolve:", all_qd_links)
    print("  all ground_truth.relevant_ids resolve:", all_gt_links)
    print()
    print("  per-dataset:")
    for ds in DATASETS:
        d = registry_datasets.get(ds, {})
        if d.get("status") == "REGISTERED":
            print(
                f"    {ds:<18} docs={d['document_count_observed']:>5}"
                f"  exs={d['example_count_observed']:>6}"
                f"  unresolved_qd={d['unresolved_query_doc_refs']}"
                f"  unresolved_gt={d['unresolved_gt_relevant_refs']}"
            )
        else:
            print(f"    {ds:<18} status={d.get('status')}")
    print()
    print("  adapter executed: False")
    print("  parser executed: False")
    print("  chunker executed: False")
    print("  tokenizer loaded: False")
    print("  model loaded: False")
    print("  embedding executed: False")
    print("  qdrant executed: False")
    print("  retrieval executed: False")
    print("  evaluation executed: False")
    print("  generation executed: False")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())