"""Clerk benchmark adapter for document-level retrieval evaluation."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sah_rag.canonical.paths import resolve_dataset_paths
from evaluation.models import (
    RetrievalEvaluationCase,
    RetrievalJudgment,
)
from sah_rag.retrieval.models import RetrievedChunk


@dataclass(frozen=True, slots=True)
class ClerkQuery:
    """One validated Clerk benchmark query."""

    query_id: str
    query_text: str
    gold_passage_ids: tuple[str, ...]
    gold_answer: str | None
    split: str | None


@dataclass(frozen=True, slots=True)
class ClerkRetrievalDataset:
    """Validated Clerk queries and document-level qrels."""

    queries: tuple[ClerkQuery, ...]
    qrels: dict[str, tuple[RetrievalJudgment, ...]]
    document_ids: frozenset[str]


def load_clerk_retrieval_dataset() -> ClerkRetrievalDataset:
    """Load and cross-validate Clerk benchmark retrieval data."""
    dataset_paths = resolve_dataset_paths("clerk")
    root = Path(dataset_paths.documents).parent

    documents_path = Path(dataset_paths.documents)
    benchmark_path = root / "benchmark.jsonl"
    qrels_path = root / "qrels.tsv"

    documents = _read_jsonl(documents_path)
    benchmark = _read_jsonl(benchmark_path)

    document_ids = frozenset(
        _required_string(record, "doc_id")
        for record in documents
    )

    if len(document_ids) != len(documents):
        raise ValueError(
            "Clerk documents contain duplicate doc_id values."
        )

    queries = tuple(
        _parse_query(record)
        for record in benchmark
    )

    query_ids = [
        query.query_id
        for query in queries
    ]

    if len(query_ids) != len(set(query_ids)):
        raise ValueError(
            "Clerk benchmark contains duplicate query_id values."
        )

    qrels = _read_qrels(qrels_path)

    benchmark_query_ids = set(query_ids)
    qrel_query_ids = set(qrels)

    if benchmark_query_ids != qrel_query_ids:
        raise ValueError(
            "Clerk benchmark and qrels query IDs do not match."
        )

    judged_document_ids = {
        judgment.item_id
        for judgments in qrels.values()
        for judgment in judgments
    }

    missing_document_ids = (
        judged_document_ids - document_ids
    )

    if missing_document_ids:
        sample = sorted(missing_document_ids)[:5]

        raise ValueError(
            "Clerk qrels reference documents missing from "
            f"the canonical corpus: {sample}"
        )

    for query in queries:
        judged_positive_ids = {
            judgment.item_id
            for judgment in qrels[query.query_id]
            if judgment.relevance > 0
        }

        benchmark_gold_ids = set(
            query.gold_passage_ids
        )

        if benchmark_gold_ids != judged_positive_ids:
            raise ValueError(
                "Clerk gold_passage_ids do not match positive "
                f"qrels for query {query.query_id}."
            )

    return ClerkRetrievalDataset(
        queries=queries,
        qrels=qrels,
        document_ids=document_ids,
    )


def rank_unique_documents(
    chunks: tuple[RetrievedChunk, ...],
    *,
    limit: int | None = None,
) -> tuple[str, ...]:
    """Convert ranked child chunks into unique document IDs."""
    if limit is not None and limit < 1:
        raise ValueError(
            "Document ranking limit must be positive."
        )

    document_ids: list[str] = []
    seen: set[str] = set()

    for chunk in sorted(
        chunks,
        key=lambda result: (
            result.rank,
            -result.score,
            result.chunk_id,
        ),
    ):
        if chunk.doc_id in seen:
            continue

        seen.add(chunk.doc_id)
        document_ids.append(chunk.doc_id)

        if (
            limit is not None
            and len(document_ids) >= limit
        ):
            break

    return tuple(document_ids)


def build_clerk_evaluation_case(
    *,
    query: ClerkQuery,
    judgments: tuple[RetrievalJudgment, ...],
    retrieved_chunks: tuple[RetrievedChunk, ...],
    retrieval_method: str,
    document_limit: int | None = None,
) -> RetrievalEvaluationCase:
    """Create one document-level Clerk evaluation case."""
    ranked_document_ids = rank_unique_documents(
        retrieved_chunks,
        limit=document_limit,
    )

    return RetrievalEvaluationCase(
        query_id=query.query_id,
        query_text=query.query_text,
        retrieval_method=retrieval_method,
        evaluation_level="document",
        judgments=judgments,
        retrieved_ids=ranked_document_ids,
        metadata={
            "dataset": "clerk",
            "split": query.split,
            "gold_passage_count": len(
                query.gold_passage_ids
            ),
            "gold_answer_available": (
                query.gold_answer is not None
            ),
            "retrieved_chunk_count": len(
                retrieved_chunks
            ),
            "retrieved_document_count": len(
                ranked_document_ids
            ),
        },
    )


def _read_jsonl(
    path: Path,
) -> list[dict[str, Any]]:
    """Read JSONL object records."""
    records: list[dict[str, Any]] = []

    with path.open(
        "r",
        encoding="utf-8-sig",
    ) as handle:
        for line_number, line in enumerate(
            handle,
            start=1,
        ):
            stripped = line.strip()

            if not stripped:
                continue

            value = json.loads(stripped)

            if not isinstance(value, dict):
                raise TypeError(
                    f"Record {line_number} in {path} "
                    "is not a JSON object."
                )

            records.append(value)

    return records


def _parse_query(
    record: dict[str, Any],
) -> ClerkQuery:
    """Parse one Clerk benchmark query."""
    query_id = _required_string(
        record,
        "query_id",
    )

    query_text = _required_string(
        record,
        "query",
    )

    raw_gold_ids = record.get(
        "gold_passage_ids"
    )

    if not isinstance(raw_gold_ids, list):
        raise TypeError(
            "Clerk gold_passage_ids must be a list."
        )

    gold_passage_ids = tuple(
        str(value)
        for value in raw_gold_ids
        if str(value).strip()
    )

    if not gold_passage_ids:
        raise ValueError(
            f"Clerk query {query_id} has no gold passages."
        )

    if len(gold_passage_ids) != len(
        set(gold_passage_ids)
    ):
        raise ValueError(
            f"Clerk query {query_id} has duplicate "
            "gold passage IDs."
        )

    raw_answer = record.get("gold_answer")
    raw_split = record.get("split")

    return ClerkQuery(
        query_id=query_id,
        query_text=query_text,
        gold_passage_ids=gold_passage_ids,
        gold_answer=(
            str(raw_answer)
            if raw_answer is not None
            else None
        ),
        split=(
            str(raw_split)
            if raw_split is not None
            else None
        ),
    )


def _read_qrels(
    path: Path,
) -> dict[str, tuple[RetrievalJudgment, ...]]:
    """Read Clerk TSV relevance judgments."""
    grouped: dict[
        str,
        list[RetrievalJudgment],
    ] = defaultdict(list)

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        reader = csv.DictReader(
            handle,
            delimiter="\t",
        )

        required_fields = {
            "query_id",
            "passage_id",
            "relevance",
        }

        if (
            reader.fieldnames is None
            or not required_fields.issubset(
                reader.fieldnames
            )
        ):
            raise ValueError(
                "Clerk qrels header is invalid."
            )

        for row_number, row in enumerate(
            reader,
            start=2,
        ):
            query_id = row["query_id"].strip()
            passage_id = row["passage_id"].strip()

            if not query_id or not passage_id:
                raise ValueError(
                    f"Empty qrels ID at row {row_number}."
                )

            try:
                relevance = int(
                    float(row["relevance"])
                )
            except ValueError as exc:
                raise ValueError(
                    "Invalid qrels relevance at row "
                    f"{row_number}: {row['relevance']}"
                ) from exc

            grouped[query_id].append(
                RetrievalJudgment(
                    item_id=passage_id,
                    relevance=relevance,
                )
            )

    result: dict[
        str,
        tuple[RetrievalJudgment, ...],
    ] = {}

    for query_id, judgments in grouped.items():
        item_ids = [
            judgment.item_id
            for judgment in judgments
        ]

        if len(item_ids) != len(set(item_ids)):
            raise ValueError(
                "Clerk qrels contain duplicate judgments "
                f"for query {query_id}."
            )

        result[query_id] = tuple(
            sorted(
                judgments,
                key=lambda judgment: (
                    judgment.item_id
                ),
            )
        )

    return result


def _required_string(
    record: dict[str, Any],
    field_name: str,
) -> str:
    """Read one required non-empty string field."""
    value = record.get(field_name)

    if not isinstance(value, str):
        raise TypeError(
            f"Required field {field_name} is not a string."
        )

    stripped = value.strip()

    if not stripped:
        raise ValueError(
            f"Required field {field_name} is empty."
        )

    return stripped


def aggregate_clerk_document_chunks(
    chunks: tuple[RetrievedChunk, ...],
    *,
    additional_chunk_weight: float,
    limit: int | None = None,
) -> tuple[RetrievedChunk, ...]:
    """Aggregate multiple retrieved chunks per Clerk document."""
    if additional_chunk_weight < 0:
        raise ValueError("additional_chunk_weight cannot be negative.")
    if limit is not None and limit < 1:
        raise ValueError("Document limit must be positive.")

    grouped: dict[str, list[RetrievedChunk]] = defaultdict(list)
    for chunk in chunks:
        grouped[chunk.doc_id].append(chunk)

    scored_documents: list[
        tuple[float, int, str, RetrievedChunk, tuple[str, ...]]
    ] = []

    for doc_id, document_chunks in grouped.items():
        ordered = sorted(
            document_chunks,
            key=lambda chunk: (chunk.rank, -chunk.score, chunk.chunk_id),
        )
        score_ordered = sorted(
            document_chunks,
            key=lambda chunk: (-chunk.score, chunk.rank, chunk.chunk_id),
        )

        representative = ordered[0]
        best_score = score_ordered[0].score
        additional_score = sum(chunk.score for chunk in score_ordered[1:])
        aggregate_score = best_score + additional_chunk_weight * additional_score

        scored_documents.append(
            (
                aggregate_score,
                representative.rank,
                doc_id,
                representative,
                tuple(chunk.chunk_id for chunk in ordered),
            )
        )

    scored_documents.sort(key=lambda item: (-item[0], item[1], item[2]))
    if limit is not None:
        scored_documents = scored_documents[:limit]

    return tuple(
        representative.model_copy(
            update={
                "rank": rank,
                "score": aggregate_score,
                "retrieval_method": "hybrid_rrf_document_aggregation",
                "payload": {
                    **representative.payload,
                    "document_chunk_ids": list(chunk_ids),
                    "document_chunk_count": len(chunk_ids),
                    "initial_best_chunk_rank": initial_rank,
                    "additional_chunk_weight": additional_chunk_weight,
                },
            }
        )
        for rank, (aggregate_score, initial_rank, _, representative, chunk_ids) in enumerate(
            scored_documents, start=1
        )
    )
