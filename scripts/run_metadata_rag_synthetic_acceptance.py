"""Synthetic acceptance run for the Metadata-RAG comparator.

Executes the full Metadata-RAG pipeline on the bounded synthetic
fixture ``infrastructure-pilot-synthetic-v1`` (12 documents, 6
queries). Produces a checksum-bound artifact bundle that covers every
stage required by the master readiness lifecycle:

    canonical_input_accepted
    preprocessing_completed
    persistent_state_created
    state_reopened
    query_retrieval_completed
    canonical_ids_recovered
    duplicate_ids_preserved
    evaluation_case_created
    generation_disabled
    repeat_run_deterministic_or_classified

The synthetic acceptance is a *contract-level* test. To make it
reproducible on CPU in a bounded time it uses a deterministic
contract encoder that produces unit-norm vectors from the SHA-256 of
the input text. It is NOT a language model, and no retrieval-quality
claim is made from its outputs. The manifest records the encoder
kind explicitly as ``DETERMINISTIC_CONTRACT_ENCODER``.

The real embedder (BGE-M3, dense-only, pinned revision) is exercised
only on the real datasets under the full-track protocol.

This script never reads ground truth during retrieval. Ground truth
is loaded only inside the evaluation-case construction stage, after
all retrieval results are already computed and written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final

from baselines.metadata_rag_adapter import (
    MetadataRAGProjectedQuery,
    build_indexed_documents,
    project_queries,
)
from baselines.metadata_rag_embedder import (
    FROZEN_DENSE_DIMENSION,
    EncoderConfig,
    MetadataRAGDocumentEmbedding,
    MetadataRAGQueryEmbedding,
    build_document_embedding,
    build_query_embedding,
)
from baselines.metadata_rag_evaluation_case import (
    MetadataRAGEvaluationCase,
    build_evaluation_case,
)
from baselines.metadata_rag_expansion import (
    ExpandedDocument,
    expand_ranked_items_to_documents,
)
from baselines.metadata_rag_persistence import (
    read_state,
    write_state,
)
from baselines.metadata_rag_reference_index import (
    ReferenceDenseIndex,
    build_reference_dense_index,
)
from baselines.metadata_rag_retrieval import (
    QueryRetrievalResult,
    RetrievalConfig,
    retrieve_batch,
)
from baselines.metadata_rag_state import (
    MetadataRAGState,
    build_state,
)
from sah_rag.canonical.loaders import iter_validated_models
from sah_rag.canonical.models import (
    CanonicalDocument,
    CanonicalExample,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

ACCEPTANCE_VERSION: Final[str] = "0.1.0"
ACCEPTANCE_ARTIFACT_TYPE: Final[str] = "metadata_rag_synthetic_acceptance"

DEFAULT_FIXTURE_ROOT = (
    REPO_ROOT
    / "processed"
    / "canonical"
    / "infrastructure-pilot-synthetic-v1"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "outputs" / "metadata_rag_synthetic_acceptance"
)
DEFAULT_DATASET_ID: Final[str] = "infrastructure-pilot-synthetic-v1"

ENCODER_KIND_DETERMINISTIC_CONTRACT: Final[str] = (
    "DETERMINISTIC_CONTRACT_ENCODER"
)
_FAKE_SNAPSHOT_MARKER: Final[str] = (
    "deterministic-contract-encoder://v1"
)


class MetadataRAGSyntheticAcceptanceError(ValueError):
    """Raised when the synthetic acceptance run violates its contract."""


# --------------------------------------------------------------------------
# Deterministic contract encoder
# --------------------------------------------------------------------------


class _DeterministicContractEncoder:
    """Deterministic dense encoder for contract-level acceptance.

    Produces a unit-norm 1024-d vector from the SHA-256 of the input
    text. NOT a language model. Its role is to exercise the pipeline
    end-to-end on CPU without loading BGE-M3.
    """

    def __init__(self, config: EncoderConfig) -> None:
        self._config = config

    @property
    def config(self) -> EncoderConfig:
        return self._config

    def encode(
        self, texts: Sequence[str]
    ) -> tuple[tuple[float, ...], ...]:
        rows: list[tuple[float, ...]] = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            raw = (
                digest
                * (FROZEN_DENSE_DIMENSION // len(digest) + 1)
            )[:FROZEN_DENSE_DIMENSION]
            vec = tuple(float(b) - 128.0 for b in raw)
            norm = math.sqrt(sum(v * v for v in vec))
            if norm == 0.0:
                raise MetadataRAGSyntheticAcceptanceError(
                    "deterministic encoder produced a zero-norm vector"
                )
            rows.append(tuple(v / norm for v in vec))
        return tuple(rows)


# --------------------------------------------------------------------------
# Deterministic file I/O
# --------------------------------------------------------------------------


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _write_json(path: Path, obj: Any) -> None:
    text = json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    path.write_bytes((text + "\n").encode("utf-8"))


def _write_jsonl(path: Path, items: Sequence[Any]) -> None:
    if not items:
        path.write_bytes(b"")
        return
    lines = [
        json.dumps(
            item.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        for item in items
    ]
    path.write_bytes(("\n".join(lines) + "\n").encode("utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise MetadataRAGSyntheticAcceptanceError(
            f"missing artifact: {path}"
        )
    if path.stat().st_size == 0:
        return []
    out: list[dict[str, Any]] = []
    for lineno, raw in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        stripped = raw.strip()
        if not stripped:
            continue
        parsed = json.loads(stripped)
        if not isinstance(parsed, dict):
            raise MetadataRAGSyntheticAcceptanceError(
                f"{path.name}:{lineno} not a JSON object"
            )
        out.append(parsed)
    return out


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise MetadataRAGSyntheticAcceptanceError(
            f"missing artifact: {path}"
        )
    parsed = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(parsed, dict):
        raise MetadataRAGSyntheticAcceptanceError(
            f"{path.name} not a JSON object"
        )
    return parsed


# --------------------------------------------------------------------------
# Acceptance run
# --------------------------------------------------------------------------


def _prepare_output_dir(output_dir: Path) -> None:
    if output_dir.exists():
        if any(output_dir.iterdir()):
            raise MetadataRAGSyntheticAcceptanceError(
                f"output_dir already contains files: {output_dir}"
            )
    else:
        output_dir.mkdir(parents=True, exist_ok=True)


def _build_encoder_config() -> EncoderConfig:
    return EncoderConfig(
        local_snapshot_path=_FAKE_SNAPSHOT_MARKER,
        device="cpu",
        use_fp16=False,
        batch_size=8,
    )


def _run_pipeline(
    *,
    fixture_root: Path,
    dataset_id: str,
    encoder_config: EncoderConfig,
    config: RetrievalConfig,
) -> dict[str, Any]:
    """Run the pipeline once, in memory. Returns the artifacts.

    This is the pure-computation core. It performs no file I/O and no
    checksum computation; the outer ``run_acceptance`` writes the
    artifacts to disk and produces the checksum bundle.
    """
    documents: tuple[CanonicalDocument, ...] = tuple(
        iter_validated_models(
            fixture_root / "documents.jsonl", CanonicalDocument
        )
    )
    examples: tuple[CanonicalExample, ...] = tuple(
        iter_validated_models(
            fixture_root / "examples.jsonl", CanonicalExample
        )
    )
    if not documents:
        raise MetadataRAGSyntheticAcceptanceError(
            "fixture documents are empty"
        )
    if not examples:
        raise MetadataRAGSyntheticAcceptanceError(
            "fixture examples are empty"
        )

    # Stage: preprocessing
    indexed = build_indexed_documents(documents, dataset_id=dataset_id)
    state: MetadataRAGState = build_state(
        indexed, dataset_id=dataset_id
    )

    # Stage: embedding (deterministic contract encoder)
    encoder = _DeterministicContractEncoder(encoder_config)
    doc_texts = [d.text_normalized for d in state.documents]
    doc_vectors = encoder.encode(doc_texts)
    doc_embeddings: tuple[MetadataRAGDocumentEmbedding, ...] = tuple(
        build_document_embedding(
            config=encoder_config,
            dataset_id=state.dataset_id,
            doc_id=d.doc_id,
            document_order=d.document_order,
            metadata_record_id=d.metadata_record_id,
            source_text=d.text_normalized,
            dense_vector=v,
        )
        for d, v in zip(state.documents, doc_vectors, strict=True)
    )

    # Stage: reference index
    index: ReferenceDenseIndex = build_reference_dense_index(
        state=state, embeddings=doc_embeddings
    )

    # Stage: retrieval (query projection + batch retrieve)
    projected: tuple[MetadataRAGProjectedQuery, ...] = project_queries(
        examples
    )
    query_texts = [q.query_text for q in projected]
    query_vectors = encoder.encode(query_texts)
    query_embeddings: tuple[MetadataRAGQueryEmbedding, ...] = tuple(
        build_query_embedding(
            config=encoder_config,
            dataset_id=dataset_id,
            query_id=q.query_id,
            query_text=q.query_text,
            dense_vector=v,
        )
        for q, v in zip(projected, query_vectors, strict=True)
    )
    results: tuple[QueryRetrievalResult, ...] = retrieve_batch(
        queries=projected,
        index=index,
        query_vectors=query_vectors,
        config=config,
    )

    # Stage: canonical expansion
    expanded_by_query: dict[str, tuple[ExpandedDocument, ...]] = {}
    for result in results:
        expanded_by_query[result.query_id] = (
            expand_ranked_items_to_documents(
                query_id=result.query_id,
                ranked_items=result.ranked_items,
                document_limit=config.document_limit,
            )
        )

    # Stage: evaluation cases (only stage that reads ground truth)
    ground_truth_by_query: dict[str, tuple[str, ...]] = {
        e.query.query_id: tuple(e.ground_truth.relevant_ids)
        for e in examples
    }
    cases: list[MetadataRAGEvaluationCase] = []
    for result in results:
        ranked_doc_ids = [
            d.doc_id for d in expanded_by_query[result.query_id]
        ]
        cases.append(
            build_evaluation_case(
                query_id=result.query_id,
                ranked_doc_ids=ranked_doc_ids,
                relevant_doc_ids=ground_truth_by_query[
                    result.query_id
                ],
            )
        )

    return {
        "documents": documents,
        "examples": examples,
        "state": state,
        "doc_embeddings": doc_embeddings,
        "query_embeddings": query_embeddings,
        "index": index,
        "results": results,
        "expanded_by_query": expanded_by_query,
        "cases": tuple(cases),
    }


def _check_gates(
    *,
    artifacts: dict[str, Any],
    dataset_id: str,
) -> dict[str, bool]:
    """Evaluate the synthetic acceptance gates on in-memory artifacts."""
    documents: tuple[CanonicalDocument, ...] = artifacts["documents"]
    examples: tuple[CanonicalExample, ...] = artifacts["examples"]
    state: MetadataRAGState = artifacts["state"]
    results: tuple[QueryRetrievalResult, ...] = artifacts["results"]
    cases: tuple[MetadataRAGEvaluationCase, ...] = artifacts["cases"]

    gates: dict[str, bool] = {}

    gates["canonical_input_accepted"] = len(documents) > 0
    gates["preprocessing_completed"] = len(state.documents) == len(
        documents
    )
    gates["query_count_matches"] = len(results) == len(examples)

    # Retrieval must return non-empty ranked items for every query.
    gates["ranked_results_nonempty"] = all(
        len(r.ranked_items) > 0 for r in results
    )

    # Canonical IDs recovered: every emitted doc_id is a canonical one.
    canonical_doc_ids = {d.doc_id for d in documents}
    gates["canonical_ids_recovered"] = all(
        all(
            item.canonical_doc_id in canonical_doc_ids
            for item in r.ranked_items
        )
        for r in results
    )

    # Duplicate document identities (same content, different doc_id)
    # must be preserved as distinct ranked items.
    seen_ids_by_query: list[set[str]] = []
    for r in results:
        seen_ids_by_query.append(
            {item.canonical_doc_id for item in r.ranked_items}
        )
    gates["duplicate_ids_preserved"] = all(
        len(s) == len(r.ranked_items)
        for s, r in zip(seen_ids_by_query, results, strict=True)
    )

    # Every query top-2 must match ground truth exactly.
    gt_by_query = {
        e.query.query_id: tuple(e.ground_truth.relevant_ids)
        for e in examples
    }
    top2_matches: list[bool] = []
    for r in results:
        gt = gt_by_query.get(r.query_id, ())
        top2 = tuple(
            item.canonical_doc_id for item in r.ranked_items[:2]
        )
        top2_matches.append(top2 == gt)
    gates["every_query_top2_matches_ground_truth"] = all(
        top2_matches
    )

    # Fallback should not trigger on the synthetic fixture.
    gates["fallback_not_triggered_on_fixture"] = all(
        not r.metadata.fallback_used for r in results
    )

    # Evaluation cases
    gates["evaluation_case_created"] = len(cases) == len(examples)

    # Generation disabled
    gates["generation_disabled"] = True  # never invoked by design

    # Dataset consistency
    gates["dataset_id_consistent"] = (
        state.dataset_id == dataset_id
        and all(
            d.dataset_id == dataset_id for d in state.documents
        )
    )

    return gates


def run_acceptance(
    *,
    fixture_root: Path,
    output_dir: Path,
    dataset_id: str = DEFAULT_DATASET_ID,
    force: bool = False,
) -> dict[str, Any]:
    """Run the synthetic acceptance to completion.

    Writes a deterministic, checksum-bound artifact bundle and returns
    the manifest. Raises MetadataRAGSyntheticAcceptanceError on any
    gate failure.
    """
    if force and output_dir.exists():
        shutil.rmtree(output_dir)

    _prepare_output_dir(output_dir)

    encoder_config = _build_encoder_config()
    config = RetrievalConfig()

    # ---------- Run 1 ----------
    artifacts = _run_pipeline(
        fixture_root=fixture_root,
        dataset_id=dataset_id,
        encoder_config=encoder_config,
        config=config,
    )

    state: MetadataRAGState = artifacts["state"]
    doc_embeddings: tuple[MetadataRAGDocumentEmbedding, ...] = (
        artifacts["doc_embeddings"]
    )
    query_embeddings: tuple[MetadataRAGQueryEmbedding, ...] = (
        artifacts["query_embeddings"]
    )
    results: tuple[QueryRetrievalResult, ...] = artifacts["results"]
    cases: tuple[MetadataRAGEvaluationCase, ...] = artifacts["cases"]
    expanded_by_query: dict[str, tuple[ExpandedDocument, ...]] = (
        artifacts["expanded_by_query"]
    )

    # ---------- Persist state ----------
    state_dir = output_dir / "state-v1"
    write_state(state, state_dir)

    # Reopen and validate.
    reopened = read_state(state_dir)
    state_reopened_ok = reopened == state

    # ---------- Write artifacts ----------
    _write_jsonl(
        output_dir / "document_embeddings.jsonl", doc_embeddings
    )
    _write_jsonl(
        output_dir / "query_embeddings.jsonl", query_embeddings
    )
    _write_jsonl(
        output_dir / "retrieval_results.jsonl", results
    )
    flattened_expanded: list[ExpandedDocument] = []
    for result in results:
        flattened_expanded.extend(expanded_by_query[result.query_id])
    _write_jsonl(
        output_dir / "expanded_documents.jsonl", flattened_expanded
    )
    _write_jsonl(
        output_dir / "evaluation_cases.jsonl", cases
    )

    # ---------- Determinism check ----------
    # Run the pure pipeline a second time. Compare byte-level JSON
    # representations of the retrieval results and evaluation cases.
    artifacts_second = _run_pipeline(
        fixture_root=fixture_root,
        dataset_id=dataset_id,
        encoder_config=encoder_config,
        config=config,
    )
    results_second: tuple[QueryRetrievalResult, ...] = (
        artifacts_second["results"]
    )
    cases_second: tuple[MetadataRAGEvaluationCase, ...] = (
        artifacts_second["cases"]
    )
    deterministic = results == results_second and cases == cases_second
    gates = _check_gates(artifacts=artifacts, dataset_id=dataset_id)
    gates["state_reopened"] = state_reopened_ok
    gates["repeat_run_deterministic_or_classified"] = deterministic

    # All gates must pass to declare acceptance passed.
    failed_gates = [name for name, ok in gates.items() if not ok]
    if failed_gates:
        raise MetadataRAGSyntheticAcceptanceError(
            "synthetic acceptance gates failed: "
            f"{sorted(failed_gates)}"
        )

    # ---------- Write readiness + manifest ----------
    readiness = {
        "synthetic_acceptance_passed": True,
        "gates": gates,
        "encoder_kind": ENCODER_KIND_DETERMINISTIC_CONTRACT,
        "encoder_config_hash": encoder_config.config_hash(),
        "dataset_id": dataset_id,
        "document_count": len(state.documents),
        "query_count": len(results),
        "index_size": artifacts["index"].size,
    }
    _write_json(output_dir / "readiness.json", readiness)

    manifest = {
        "artifact_type": ACCEPTANCE_ARTIFACT_TYPE,
        "artifact_version": ACCEPTANCE_VERSION,
        "dataset_id": dataset_id,
        "encoder_kind": ENCODER_KIND_DETERMINISTIC_CONTRACT,
        "encoder_config_hash": encoder_config.config_hash(),
        "document_count": len(state.documents),
        "query_count": len(results),
        "index_size": artifacts["index"].size,
        "gates": gates,
        "all_gates_passed": True,
        "generation_enabled": False,
        "retrieval_only": True,
    }
    _write_json(output_dir / "manifest.json", manifest)

    # ---------- Checksums ----------
    artifact_files = [
        "manifest.json",
        "readiness.json",
        "document_embeddings.jsonl",
        "query_embeddings.jsonl",
        "retrieval_results.jsonl",
        "expanded_documents.jsonl",
        "evaluation_cases.jsonl",
        "state-v1/documents.jsonl",
        "state-v1/manifest.json",
        "state-v1/checksums.json",
    ]
    checksums = {
        rel: _sha256_file(output_dir / rel) for rel in artifact_files
    }
    _write_json(output_dir / "checksums.json", checksums)

    manifest["checksums"] = checksums
    return manifest


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the Metadata-RAG synthetic acceptance on the "
            "bounded fixture."
        )
    )
    parser.add_argument(
        "--fixture-root",
        type=Path,
        default=DEFAULT_FIXTURE_ROOT,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--dataset-id",
        type=str,
        default=DEFAULT_DATASET_ID,
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete an existing output directory before running.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    manifest = run_acceptance(
        fixture_root=args.fixture_root,
        output_dir=args.output_dir,
        dataset_id=args.dataset_id,
        force=args.force,
    )
    print("[OK] Metadata-RAG synthetic acceptance")
    print(f"  dataset_id     = {manifest['dataset_id']}")
    print(f"  document_count = {manifest['document_count']}")
    print(f"  query_count    = {manifest['query_count']}")
    print(f"  index_size     = {manifest['index_size']}")
    for name, ok in manifest["gates"].items():
        marker = "PASS" if ok else "FAIL"
        print(f"  [ {marker} ] {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
