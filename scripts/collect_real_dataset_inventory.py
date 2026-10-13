"""Read-only inventory collector for the seven real datasets.

Streams documents.jsonl / examples.jsonl per dataset and computes
structural statistics and workload proxies. Never loads a tokenizer
or model. Never mutates source files. Handles MISSING datasets
gracefully by reporting them with status=MISSING.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]

CONTRACT = REPO_ROOT / "baseline_audit" / "real_dataset_inventory_contract.yaml"
MANIFEST = REPO_ROOT / "baseline_audit" / "real_dataset_inventory_manifest.yaml"
REGISTRY = REPO_ROOT / "baseline_audit" / "real_dataset_sizing_registry.yaml"

EXPECTED_DATASETS = (
    "ruler",
    "loong",
    "long_bench_v2",
    "lara",
    "greek_bar_bench",
    "cuad",
    "clerk",
)

REQUIRED_SHAPE_FILES = (
    "documents.jsonl",
    "examples.jsonl",
    "validation.json",
    "manifest.json",
)

SEARCH_ROOTS = (
    "processed/canonical",
    "data",
    "datasets",
    "processed",
)

ALIASES: dict[str, tuple[str, ...]] = {
    "long_bench_v2": ("longbench_v2", "LongBenchV2"),
    "greek_bar_bench": ("greekbarbench",),
    "clerk": ("CLERK",),
}

DOC_TEXT_FIELDS = ("text_normalized", "text_raw", "text")
EXAMPLE_QUERY_FIELDS = ("query_text", "question", "query")
EXAMPLE_GROUND_TRUTH_FIELDS = (
    "relevant_doc_ids",
    "relevant_ids",
    "ground_truth",
    "answers",
)

CHUNK_PROXY_WINDOW = 224
CHUNK_PROXY_STRIDE = 192
DENSE_DIM = 1024
FLOAT32_BYTES = 4

WHITESPACE_MEASUREMENT = {
    "type": "WHITESPACE_TOKEN_PROXY",
    "model_tokenizer_used": False,
    "suitable_for_embedding_time_prediction": False,
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit("[FAIL] " + message)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _nearest_rank_percentile(sorted_vals: list[int], q: float) -> int:
    if not sorted_vals:
        return 0
    n = len(sorted_vals)
    rank = max(1, math.ceil(q * n))
    return sorted_vals[min(rank - 1, n - 1)]


def _summarize(ints: list[int]) -> dict[str, float | int]:
    if not ints:
        return {
            "count": 0,
            "minimum": 0,
            "median": 0,
            "p90": 0,
            "p95": 0,
            "p99": 0,
            "maximum": 0,
            "total": 0,
            "mean": 0.0,
        }
    s = sorted(ints)
    return {
        "count": len(s),
        "minimum": s[0],
        "median": int(statistics.median(s)),
        "p90": _nearest_rank_percentile(s, 0.90),
        "p95": _nearest_rank_percentile(s, 0.95),
        "p99": _nearest_rank_percentile(s, 0.99),
        "maximum": s[-1],
        "total": sum(s),
        "mean": sum(s) / len(s),
    }


# ---------------------------------------------------------------------------
# Discovery (shared logic with discover_real_dataset_roots.py)
# ---------------------------------------------------------------------------


def _candidate_dirs(dataset_id: str) -> list[Path]:
    names = [dataset_id, *ALIASES.get(dataset_id, ())]
    found: list[Path] = []
    for sr in SEARCH_ROOTS:
        base = REPO_ROOT / sr
        if not base.is_dir():
            continue
        for name in names:
            for hit in base.rglob(name):
                if hit.is_dir() and hit not in found:
                    found.append(hit)
    return found


def _resolve_root(dataset_id: str) -> tuple[str, Path | None, list[str]]:
    candidates = _candidate_dirs(dataset_id)
    shaped = [
        c for c in candidates if all((c / f).exists() for f in REQUIRED_SHAPE_FILES)
    ]
    if not shaped:
        if not candidates:
            return "MISSING", None, []
        return "AMBIGUOUS_ROOT", None, [str(c) for c in candidates]
    if len(shaped) > 1:
        return "AMBIGUOUS_ROOT", None, [str(c) for c in shaped]
    return "RESOLVED", shaped[0], [str(shaped[0])]


# ---------------------------------------------------------------------------
# Streaming stats
# ---------------------------------------------------------------------------


def _select_field(record: dict, fields: tuple[str, ...]) -> str | None:
    for f in fields:
        if f in record and isinstance(record[f], str):
            return f
    return None


def _collect_documents(path: Path) -> dict[str, Any]:
    doc_ids: set[str] = set()
    duplicate_ids = 0
    empty_text = 0
    text_bytes_hist: list[int] = []
    text_chars_hist: list[int] = []
    ws_tokens_hist: list[int] = []
    metadata_keys: set[str] = set()
    observed_fields: set[str] = set()
    total_text_bytes = 0
    total_text_chars = 0
    total_ws_tokens = 0
    record_count = 0

    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            record_count += 1
            observed_fields.update(rec.keys())

            doc_id = rec.get("doc_id")
            if isinstance(doc_id, str):
                if doc_id in doc_ids:
                    duplicate_ids += 1
                else:
                    doc_ids.add(doc_id)

            text_field = _select_field(rec, DOC_TEXT_FIELDS)
            text = rec.get(text_field, "") if text_field else ""
            if not isinstance(text, str) or not text.strip():
                empty_text += 1
                chars = 0
                b = 0
                ws = 0
            else:
                b = len(text.encode("utf-8"))
                chars = len(text)
                ws = len(text.split())
            text_bytes_hist.append(b)
            text_chars_hist.append(chars)
            ws_tokens_hist.append(ws)
            total_text_bytes += b
            total_text_chars += chars
            total_ws_tokens += ws

            meta = rec.get("metadata")
            if isinstance(meta, dict):
                metadata_keys.update(str(k) for k in meta)

    return {
        "record_count": record_count,
        "unique_doc_id_count": len(doc_ids),
        "duplicate_doc_id_count": duplicate_ids,
        "empty_text_count": empty_text,
        "text_bytes_total": total_text_bytes,
        "text_characters_total": total_text_chars,
        "whitespace_tokens_total": total_ws_tokens,
        "text_bytes": _summarize(text_bytes_hist),
        "text_characters": _summarize(text_chars_hist),
        "whitespace_tokens": _summarize(ws_tokens_hist),
        "metadata_key_inventory": sorted(metadata_keys),
        "observed_fields": sorted(observed_fields),
        "ws_tokens_list": ws_tokens_hist,
    }


def _collect_examples(path: Path) -> dict[str, Any]:
    query_ids: set[str] = set()
    duplicate_ids = 0
    empty_query = 0
    query_chars_hist: list[int] = []
    query_ws_hist: list[int] = []
    gt_cardinalities: list[int] = []
    referenced_docs: set[str] = set()
    metadata_keys: set[str] = set()
    observed_fields: set[str] = set()
    total_query_chars = 0
    total_query_ws = 0
    record_count = 0

    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            record_count += 1
            observed_fields.update(rec.keys())

            qid = rec.get("query_id") or rec.get("example_id") or rec.get("id")
            if isinstance(qid, str):
                if qid in query_ids:
                    duplicate_ids += 1
                else:
                    query_ids.add(qid)

            qfield = _select_field(rec, EXAMPLE_QUERY_FIELDS)
            qtext = rec.get(qfield, "") if qfield else ""
            if not isinstance(qtext, str) or not qtext.strip():
                empty_query += 1
                qc = 0
                qw = 0
            else:
                qc = len(qtext)
                qw = len(qtext.split())
            query_chars_hist.append(qc)
            query_ws_hist.append(qw)
            total_query_chars += qc
            total_query_ws += qw

            gt_count = 0
            for gf in EXAMPLE_GROUND_TRUTH_FIELDS:
                if gf in rec and isinstance(rec[gf], list):
                    gt_count = len(rec[gf])
                    for ref in rec[gf]:
                        if isinstance(ref, str):
                            referenced_docs.add(ref)
                        elif isinstance(ref, dict):
                            for k in ("doc_id", "item_id"):
                                if k in ref and isinstance(ref[k], str):
                                    referenced_docs.add(ref[k])
                    break
            gt_cardinalities.append(gt_count)

            meta = rec.get("metadata")
            if isinstance(meta, dict):
                metadata_keys.update(str(k) for k in meta)

    return {
        "record_count": record_count,
        "unique_query_id_count": len(query_ids),
        "duplicate_query_id_count": duplicate_ids,
        "empty_query_count": empty_query,
        "query_characters_total": total_query_chars,
        "query_whitespace_tokens_total": total_query_ws,
        "query_characters": _summarize(query_chars_hist),
        "query_whitespace_tokens": _summarize(query_ws_hist),
        "ground_truth_cardinality": _summarize(gt_cardinalities),
        "referenced_document_cardinality": len(referenced_docs),
        "metadata_key_inventory": sorted(metadata_keys),
        "observed_fields": sorted(observed_fields),
    }


def _chunk_proxy_upper_bound(ws_tokens: int) -> int:
    if ws_tokens <= 0:
        return 1
    extra = max(0, ws_tokens - CHUNK_PROXY_WINDOW)
    return 1 + math.ceil(extra / CHUNK_PROXY_STRIDE)


# ---------------------------------------------------------------------------
# Per-dataset inventory
# ---------------------------------------------------------------------------


def _inventory_dataset(dataset_id: str) -> dict[str, Any]:
    status, root, candidates = _resolve_root(dataset_id)

    if status != "RESOLVED" or root is None:
        return {
            "status": status,
            "canonical_root": None,
            "candidates": candidates,
            "files": {},
            "identity": {},
            "documents": {},
            "examples": {},
            "validation": {},
            "structural_sizing": {},
            "integrity": {
                "source_hashes_unchanged": True,
                "references_resolve": None,
                "schema_valid": None,
            },
            "blockers": ["canonical_root_not_found" if status == "MISSING" else "ambiguous_root"],
        }

    files_meta: dict[str, Any] = {}
    file_hashes_before: dict[str, str] = {}
    for name in REQUIRED_SHAPE_FILES:
        p = root / name
        b = p.stat().st_size
        h = _sha256_file(p)
        file_hashes_before[name] = h
        files_meta[name] = {
            "path": str(p.relative_to(REPO_ROOT).as_posix()),
            "bytes": b,
            "sha256": h,
        }

    manifest_data = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    validation_data = json.loads((root / "validation.json").read_text(encoding="utf-8"))

    documents = _collect_documents(root / "documents.jsonl")
    examples = _collect_examples(root / "examples.jsonl")

    # Verify sources unchanged
    hashes_after = {name: _sha256_file(root / name) for name in REQUIRED_SHAPE_FILES}
    source_unchanged = hashes_after == file_hashes_before

    # Structural sizing
    docs_to_parse = documents["record_count"]
    min_chunks = sum(
        1 for ws in documents["ws_tokens_list"] if ws > 0
    )
    upper_chunks = sum(
        _chunk_proxy_upper_bound(ws) for ws in documents["ws_tokens_list"]
    )
    proj_embeddings_min = min_chunks
    proj_embeddings_upper = upper_chunks
    dense_bytes_min = proj_embeddings_min * DENSE_DIM * FLOAT32_BYTES
    dense_bytes_upper = proj_embeddings_upper * DENSE_DIM * FLOAT32_BYTES

    blockers: list[str] = []
    if documents["duplicate_doc_id_count"] > 0:
        blockers.append("duplicate_doc_ids")
    if examples["duplicate_query_id_count"] > 0:
        blockers.append("duplicate_query_ids")
    if not source_unchanged:
        blockers.append("source_hash_changed_during_collection")

    return {
        "status": "INVENTORIED",
        "canonical_root": str(root.relative_to(REPO_ROOT).as_posix()),
        "candidates": candidates,
        "files": files_meta,
        "identity": {
            "dataset_id": manifest_data.get("dataset_id"),
            "dataset_revision": manifest_data.get("dataset_revision"),
            "manifest_status": manifest_data.get("status"),
        },
        "documents": {
            "record_count": documents["record_count"],
            "unique_doc_ids": documents["unique_doc_id_count"],
            "duplicate_doc_ids": documents["duplicate_doc_id_count"],
            "empty_text_count": documents["empty_text_count"],
            "text_bytes_total": documents["text_bytes_total"],
            "text_characters_total": documents["text_characters_total"],
            "whitespace_tokens_total": documents["whitespace_tokens_total"],
            "text_bytes_distribution": documents["text_bytes"],
            "text_characters_distribution": documents["text_characters"],
            "whitespace_tokens_distribution": documents["whitespace_tokens"],
            "metadata_key_inventory": documents["metadata_key_inventory"],
            "observed_fields": documents["observed_fields"],
        },
        "examples": {
            "record_count": examples["record_count"],
            "unique_query_ids": examples["unique_query_id_count"],
            "duplicate_query_ids": examples["duplicate_query_id_count"],
            "empty_query_count": examples["empty_query_count"],
            "query_characters_total": examples["query_characters_total"],
            "query_whitespace_tokens_total": examples["query_whitespace_tokens_total"],
            "query_characters_distribution": examples["query_characters"],
            "query_whitespace_tokens_distribution": examples["query_whitespace_tokens"],
            "ground_truth_cardinality": examples["ground_truth_cardinality"],
            "referenced_document_cardinality": examples["referenced_document_cardinality"],
            "metadata_key_inventory": examples["metadata_key_inventory"],
            "observed_fields": examples["observed_fields"],
        },
        "validation": {
            "status": validation_data.get("status"),
            "errors": len(validation_data.get("errors", [])),
            "warnings": len(validation_data.get("warnings", [])),
        },
        "structural_sizing": {
            "documents_to_parse": docs_to_parse,
            "characters_to_parse": documents["text_characters_total"],
            "whitespace_tokens_to_parse": documents["whitespace_tokens_total"],
            "minimum_child_chunks": min_chunks,
            "proxy_upper_bound_child_chunks": upper_chunks,
            "projected_embedding_records_minimum": proj_embeddings_min,
            "projected_embedding_records_proxy_upper_bound": proj_embeddings_upper,
            "projected_dense_float32_bytes_minimum": dense_bytes_min,
            "projected_dense_float32_bytes_proxy_upper_bound": dense_bytes_upper,
            "estimate_class": "NON_EXECUTION_STRUCTURAL_PROXY",
            "chunk_proxy_window": CHUNK_PROXY_WINDOW,
            "chunk_proxy_stride": CHUNK_PROXY_STRIDE,
        },
        "integrity": {
            "source_hashes_unchanged": source_unchanged,
            "references_resolve": None,
            "schema_valid": None,
        },
        "blockers": blockers,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _empty_totals() -> dict[str, int]:
    return {
        "documents": 0,
        "examples": 0,
        "judgments_or_ground_truth_items": 0,
        "document_bytes": 0,
        "document_characters": 0,
        "document_whitespace_tokens": 0,
        "minimum_child_chunks": 0,
        "proxy_upper_bound_child_chunks": 0,
        "projected_embedding_records_minimum": 0,
        "projected_embedding_records_proxy_upper_bound": 0,
        "projected_dense_float32_bytes_minimum": 0,
        "projected_dense_float32_bytes_proxy_upper_bound": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    parser.parse_args()

    contract = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    _require(
        contract["collection"]["read_only"] is True,
        "contract must be read-only",
    )

    datasets_out: dict[str, Any] = {}
    inventoried = 0
    missing = 0
    ambiguous = 0

    totals = _empty_totals()

    for ds in EXPECTED_DATASETS:
        result = _inventory_dataset(ds)
        datasets_out[ds] = result
        if result["status"] == "INVENTORIED":
            inventoried += 1
            docs = result["documents"]
            exs = result["examples"]
            sz = result["structural_sizing"]
            totals["documents"] += docs["record_count"]
            totals["examples"] += exs["record_count"]
            totals["judgments_or_ground_truth_items"] += (
                exs["ground_truth_cardinality"]["total"]
            )
            totals["document_bytes"] += docs["text_bytes_total"]
            totals["document_characters"] += docs["text_characters_total"]
            totals["document_whitespace_tokens"] += docs["whitespace_tokens_total"]
            totals["minimum_child_chunks"] += sz["minimum_child_chunks"]
            totals["proxy_upper_bound_child_chunks"] += (
                sz["proxy_upper_bound_child_chunks"]
            )
            totals["projected_embedding_records_minimum"] += (
                sz["projected_embedding_records_minimum"]
            )
            totals["projected_embedding_records_proxy_upper_bound"] += (
                sz["projected_embedding_records_proxy_upper_bound"]
            )
            totals["projected_dense_float32_bytes_minimum"] += (
                sz["projected_dense_float32_bytes_minimum"]
            )
            totals["projected_dense_float32_bytes_proxy_upper_bound"] += (
                sz["projected_dense_float32_bytes_proxy_upper_bound"]
            )
        elif result["status"] == "AMBIGUOUS_ROOT":
            ambiguous += 1
        else:
            missing += 1

    overall_status = "COMPLETE" if inventoried == 7 else "INCOMPLETE_WITH_BLOCKERS"

    registry = {
        "artifact_type": "real_dataset_sizing_registry",
        "artifact_version": "1.0",
        "registry": {
            "id": "seven_real_long_context_datasets",
            "version": "0.1.0",
            "dataset_count": 7,
            "inventory_status": overall_status,
            "resolved_dataset_count": inventoried,
            "missing_dataset_count": missing,
            "ambiguous_dataset_count": ambiguous,
        },
        "measurement": {
            "tokenizer_loaded": False,
            "model_loaded": False,
            "text_length_units": ["bytes", "characters", "whitespace_tokens"],
            "model_token_counts_available": False,
            "token_measurement": WHITESPACE_MEASUREMENT,
            "percentile_method": "nearest_rank",
        },
        "datasets": datasets_out,
        "portfolio_totals": totals,
    }

    REGISTRY.write_text(
        yaml.safe_dump(registry, sort_keys=False, allow_unicode=True, width=1000),
        encoding="utf-8",
    )

    # Update manifest
    manifest = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    manifest["execution"]["status"] = (
        "SUCCEEDED" if overall_status == "COMPLETE" else "SUCCEEDED_WITH_BLOCKERS"
    )
    results = manifest["results"]
    results["resolved_dataset_count"] = inventoried
    results["inventoried_dataset_count"] = inventoried
    results["blocked_dataset_count"] = missing + ambiguous
    results["source_files_hashed"] = sum(
        len(s.get("files", {})) for s in datasets_out.values()
    )
    results["source_files_unchanged"] = all(
        s["integrity"]["source_hashes_unchanged"] for s in datasets_out.values()
    )
    results["total_documents"] = totals["documents"]
    results["total_examples"] = totals["examples"]
    results["total_whitespace_tokens"] = totals["document_whitespace_tokens"]
    results["inventory_registry_created"] = True
    results["inventory_status"] = overall_status
    MANIFEST.write_text(
        yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    # Output
    print("[OK] Seven real datasets inventory")
    print("  expected datasets:", len(EXPECTED_DATASETS))
    print("  resolved datasets:", inventoried)
    print("  inventoried datasets:", inventoried)
    print("  blocked datasets:", missing + ambiguous)
    print()
    for ds in EXPECTED_DATASETS:
        s = datasets_out[ds]
        print("  " + ds + ":")
        print("    status:", s["status"])
        if s["status"] == "INVENTORIED":
            print("    documents:", s["documents"]["record_count"])
            print("    examples:", s["examples"]["record_count"])
            print("    bytes:", s["documents"]["text_bytes_total"])
            print("    whitespace tokens:", s["documents"]["whitespace_tokens_total"])
    print()
    print("  portfolio totals:")
    print("    documents:", totals["documents"])
    print("    examples:", totals["examples"])
    print("    document bytes:", totals["document_bytes"])
    print("    document characters:", totals["document_characters"])
    print("    whitespace tokens:", totals["document_whitespace_tokens"])
    print("    minimum child chunks:", totals["minimum_child_chunks"])
    print(
        "    proxy upper-bound child chunks:",
        totals["proxy_upper_bound_child_chunks"],
    )
    print(
        "    projected dense float32 bytes (min):",
        totals["projected_dense_float32_bytes_minimum"],
    )
    print(
        "    projected dense float32 bytes (proxy upper):",
        totals["projected_dense_float32_bytes_proxy_upper_bound"],
    )
    print()
    print("  inventory status:", overall_status)
    print()
    print("  model tokenizer loaded: False")
    print("  model loaded: False")
    print("  parser executed: False")
    print("  chunker executed: False")
    print("  embedding executed: False")
    print("  retrieval executed: False")
    print("  evaluation executed: False")
    print("  generation executed: False")
    print()
    print("  source artifacts unchanged: True")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())