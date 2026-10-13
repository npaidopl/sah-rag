"""Typed synthetic pilot dataset (Step 78).

Deterministic, project-owned, text-only. No random generation, no
external reads, no network access. All content is generated from
fixed constants defined in this module.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

DATASET_ID = "infrastructure-pilot-synthetic-v1"
DATASET_REVISION = "1.0.0"
SPLIT = "pilot"

_TOPICS: tuple[dict[str, str], ...] = (
    {
        "group": "pilot-content-group-01",
        "query": (
            "Which instruments record wind direction and air-pressure "
            "changes at the Amber Observatory?"
        ),
        "para1": (
            "The Amber Observatory records atmospheric patterns from a low "
            "hill above the fictional town of Northbridge. Its eastern "
            "instrument measures wind direction, while the western instrument "
            "records changes in air pressure. Observations are copied into a "
            "public log at noon each day."
        ),
        "para2": (
            "During the pilot season, the observatory issued six weekly "
            "summaries. Each summary linked local measurements to the matching "
            "instrument record and preserved the original observation order "
            "for later review."
        ),
    },
    {
        "group": "pilot-content-group-02",
        "query": (
            "How does the Cedar Archive preserve the order of transferred "
            "records?"
        ),
        "para1": (
            "The Cedar Archive stores transferred documents in numbered "
            "shelves within a climate-controlled room. Each shelf receives "
            "records from a single collection so that order is preserved "
            "across moves. A simple index lists the shelf number, the "
            "collection name, and the arrival date."
        ),
        "para2": (
            "When a document leaves the archive, a receiving clerk signs the "
            "transfer log and notes the return deadline. The archive keeps "
            "the original accession number on every document so that later "
            "queries can trace each item back to its first catalogue entry."
        ),
    },
    {
        "group": "pilot-content-group-03",
        "query": (
            "What environmental systems does the Lunar Greenhouse monitor?"
        ),
        "para1": (
            "The Lunar Greenhouse monitors three environmental systems: "
            "humidity, temperature, and light. Sensors report each reading to "
            "a central panel every hour, and the panel compares readings "
            "against the day's target range. Growers adjust vents and lamps "
            "when a reading drifts beyond the target range."
        ),
        "para2": (
            "The greenhouse logs every adjustment together with the sensor "
            "reading that triggered it. Weekly reviews compare the adjustments "
            "against growth records so that the team can refine the "
            "environmental targets for the following cycle."
        ),
    },
    {
        "group": "pilot-content-group-04",
        "query": (
            "How does the River Signal Station indicate whether small craft "
            "may proceed?"
        ),
        "para1": (
            "The River Signal Station tracks water levels along a short "
            "stretch of the fictional Marlowe River. Two gauges report at "
            "dawn and dusk, and a third reports after heavy rain. Signal "
            "lamps on the eastern bank indicate whether small craft may "
            "proceed downstream."
        ),
        "para2": (
            "The station records each reading, the corresponding lamp state, "
            "and the time of the observation. Monthly summaries compare these "
            "records to the historic range so that unusual readings can be "
            "flagged for follow-up."
        ),
    },
    {
        "group": "pilot-content-group-05",
        "query": (
            "How does the Silver Workshop trace a tool back to its raw "
            "material batch?"
        ),
        "para1": (
            "The Silver Workshop produces small hand tools using traditional "
            "forging methods. Each tool passes through four stages: shaping, "
            "hardening, polishing, and inspection. The workshop stamps a "
            "serial number on every finished tool so that a later repair can "
            "locate the original batch."
        ),
        "para2": (
            "Batch records list the tools produced on each working day "
            "together with the raw material supplier. When a defect is "
            "reported, the workshop compares the tool serial against the "
            "batch record to identify the affected supplier shipment."
        ),
    },
    {
        "group": "pilot-content-group-06",
        "query": (
            "What does the Winter Navigation School record for each training "
            "session?"
        ),
        "para1": (
            "The Winter Navigation School trains small-craft operators in "
            "cold-weather piloting along the fictional Halden coast. Courses "
            "cover ice reading, wind awareness, and emergency signalling. "
            "Instructors record each session and note which techniques were "
            "demonstrated."
        ),
        "para2": (
            "Graduates receive a numbered certificate that lists the training "
            "sessions they completed. The school keeps a register of "
            "certificates so that a missing certificate can be reissued from "
            "the original records."
        ),
    },
)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _normalize(text: str) -> str:
    return " ".join(text.split())


def _token_count(text: str) -> int:
    return len(text.split())


def _is_sha256(value: str) -> bool:
    if len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return value == value.lower()


class PilotCanonicalDocument(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    dataset_id: str
    doc_id: str
    content_group_id: str
    text: str
    text_normalized: str
    source_text_sha256: str
    split: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator(
        "dataset_id", "doc_id", "content_group_id", "text", "text_normalized"
    )
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v:
            raise ValueError("must be non-empty")
        return v

    @field_validator("source_text_sha256")
    @classmethod
    def _sha(cls, v: str) -> str:
        if not _is_sha256(v):
            raise ValueError("must be a 64-char lowercase hex SHA-256")
        return v

    @field_validator("split")
    @classmethod
    def _split(cls, v: str) -> str:
        if v != SPLIT:
            raise ValueError(f"split must be {SPLIT}")
        return v

    @model_validator(mode="after")
    def _consistency(self) -> PilotCanonicalDocument:
        if _normalize(self.text) != self.text_normalized:
            raise ValueError("text_normalized must equal normalized text")
        if _sha256(self.text_normalized) != self.source_text_sha256:
            raise ValueError("source_text_sha256 must match text_normalized")
        return self


class PilotCanonicalChunk(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    dataset_id: str
    doc_id: str
    parent_id: str
    chunk_id: str
    content_group_id: str
    source_order: int
    text: str
    token_count: int
    source_text_sha256: str
    split: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator(
        "dataset_id", "doc_id", "parent_id", "chunk_id",
        "content_group_id", "text",
    )
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v:
            raise ValueError("must be non-empty")
        return v

    @field_validator("source_order")
    @classmethod
    def _order(cls, v: int) -> int:
        if v < 0:
            raise ValueError("source_order must be >= 0")
        return v

    @field_validator("token_count")
    @classmethod
    def _tokens(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("token_count must be > 0")
        return v

    @field_validator("source_text_sha256")
    @classmethod
    def _sha(cls, v: str) -> str:
        if not _is_sha256(v):
            raise ValueError("must be a 64-char lowercase hex SHA-256")
        return v

    @field_validator("split")
    @classmethod
    def _split(cls, v: str) -> str:
        if v != SPLIT:
            raise ValueError(f"split must be {SPLIT}")
        return v

    @model_validator(mode="after")
    def _consistency(self) -> PilotCanonicalChunk:
        if self.parent_id != self.doc_id:
            raise ValueError("parent_id must equal doc_id")
        if _sha256(self.text) != self.source_text_sha256:
            raise ValueError("source_text_sha256 must match chunk text")
        return self


class PilotCanonicalQuery(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    dataset_id: str
    query_id: str
    query_text: str
    target_content_group_id: str
    split: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator(
        "dataset_id", "query_id", "query_text", "target_content_group_id"
    )
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v:
            raise ValueError("must be non-empty")
        return v

    @field_validator("split")
    @classmethod
    def _split(cls, v: str) -> str:
        if v != SPLIT:
            raise ValueError(f"split must be {SPLIT}")
        return v


class PilotCanonicalJudgment(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    dataset_id: str
    query_id: str
    item_id: str
    relevance: int
    split: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("dataset_id", "query_id", "item_id")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v:
            raise ValueError("must be non-empty")
        return v

    @field_validator("relevance")
    @classmethod
    def _rel(cls, v: int) -> int:
        if v != 1:
            raise ValueError("pilot judgments must have relevance == 1")
        return v

    @field_validator("split")
    @classmethod
    def _split(cls, v: str) -> str:
        if v != SPLIT:
            raise ValueError(f"split must be {SPLIT}")
        return v


class PilotDatasetManifest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    artifact_type: str = "pilot_dataset_manifest"
    artifact_version: str = "1.0"

    dataset_id: str
    dataset_revision: str
    split: str

    document_count: int = Field(ge=0)
    unique_content_count: int = Field(ge=0)
    duplicate_content_pair_count: int = Field(ge=0)
    chunk_count: int = Field(ge=0)
    query_count: int = Field(ge=0)
    judgment_count: int = Field(ge=0)

    documents_path: str
    chunks_path: str
    queries_path: str
    judgments_path: str

    documents_sha256: str
    chunks_sha256: str
    queries_sha256: str
    judgments_sha256: str

    synthetic: bool = True
    project_owned: bool = True
    real_dataset: bool = False
    publishable_dataset_result: bool = False

    baseline_execution_enabled: bool = False
    model_execution_enabled: bool = False
    retrieval_execution_enabled: bool = False
    evaluation_execution_enabled: bool = False
    final_generation_enabled: bool = False

    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator(
        "documents_sha256", "chunks_sha256",
        "queries_sha256", "judgments_sha256",
    )
    @classmethod
    def _sha(cls, v: str) -> str:
        if not _is_sha256(v):
            raise ValueError("must be a 64-char lowercase hex SHA-256")
        return v

    @model_validator(mode="after")
    def _safety(self) -> PilotDatasetManifest:
        if self.real_dataset is not False:
            raise ValueError("real_dataset must be false")
        if self.publishable_dataset_result is not False:
            raise ValueError("publishable_dataset_result must be false")
        for flag in (
            self.baseline_execution_enabled,
            self.model_execution_enabled,
            self.retrieval_execution_enabled,
            self.evaluation_execution_enabled,
            self.final_generation_enabled,
        ):
            if flag is not False:
                raise ValueError("execution flags must be false")
        return self


def build_infrastructure_pilot_dataset() -> tuple[
    tuple[PilotCanonicalDocument, ...],
    tuple[PilotCanonicalChunk, ...],
    tuple[PilotCanonicalQuery, ...],
    tuple[PilotCanonicalJudgment, ...],
]:
    """Build the deterministic synthetic pilot dataset."""
    documents: list[PilotCanonicalDocument] = []
    chunks: list[PilotCanonicalChunk] = []
    queries: list[PilotCanonicalQuery] = []
    judgments: list[PilotCanonicalJudgment] = []

    for topic_idx, topic in enumerate(_TOPICS, start=1):
        group = topic["group"]
        para1 = topic["para1"]
        para2 = topic["para2"]
        full_text = para1 + "\n\n" + para2
        normalized = _normalize(full_text)
        doc_hash = _sha256(normalized)

        doc_a_num = (topic_idx - 1) * 2 + 1
        doc_b_num = (topic_idx - 1) * 2 + 2
        doc_a = f"pilot-doc-{doc_a_num:02d}"
        doc_b = f"pilot-doc-{doc_b_num:02d}"

        for doc_id, doc_num in ((doc_a, doc_a_num), (doc_b, doc_b_num)):
            documents.append(
                PilotCanonicalDocument(
                    dataset_id=DATASET_ID,
                    doc_id=doc_id,
                    content_group_id=group,
                    text=full_text,
                    text_normalized=normalized,
                    source_text_sha256=doc_hash,
                    split=SPLIT,
                    metadata={"source_topic_index": topic_idx},
                )
            )
            for chunk_idx, para in enumerate((para1, para2)):
                chunks.append(
                    PilotCanonicalChunk(
                        dataset_id=DATASET_ID,
                        doc_id=doc_id,
                        parent_id=doc_id,
                        chunk_id=f"pilot-chunk-{doc_num:02d}-{chunk_idx}",
                        content_group_id=group,
                        source_order=chunk_idx,
                        text=para,
                        token_count=_token_count(para),
                        source_text_sha256=_sha256(para),
                        split=SPLIT,
                        metadata={"source_document_sha256": doc_hash},
                    )
                )

        query_id = f"pilot-query-{topic_idx:02d}"
        queries.append(
            PilotCanonicalQuery(
                dataset_id=DATASET_ID,
                query_id=query_id,
                query_text=topic["query"],
                target_content_group_id=group,
                split=SPLIT,
                metadata={},
            )
        )
        for doc_id in (doc_a, doc_b):
            judgments.append(
                PilotCanonicalJudgment(
                    dataset_id=DATASET_ID,
                    query_id=query_id,
                    item_id=doc_id,
                    relevance=1,
                    split=SPLIT,
                    metadata={"item_type": "canonical_document"},
                )
            )

    documents.sort(key=lambda d: d.doc_id)
    chunks.sort(key=lambda c: (c.doc_id, c.source_order, c.chunk_id))
    queries.sort(key=lambda q: q.query_id)
    judgments.sort(key=lambda j: (j.query_id, j.item_id))

    return (
        tuple(documents),
        tuple(chunks),
        tuple(queries),
        tuple(judgments),
    )


def validate_infrastructure_pilot_dataset(
    *,
    documents: Sequence[PilotCanonicalDocument],
    chunks: Sequence[PilotCanonicalChunk],
    queries: Sequence[PilotCanonicalQuery],
    judgments: Sequence[PilotCanonicalJudgment],
) -> None:
    if len(documents) != 12:
        raise ValueError("expected 12 documents")
    if len(chunks) != 24:
        raise ValueError("expected 24 chunks")
    if len(queries) != 6:
        raise ValueError("expected 6 queries")
    if len(judgments) != 12:
        raise ValueError("expected 12 judgments")

    # unique IDs
    if len({d.doc_id for d in documents}) != 12:
        raise ValueError("duplicate doc_id")
    if len({c.chunk_id for c in chunks}) != 24:
        raise ValueError("duplicate chunk_id")
    if len({q.query_id for q in queries}) != 6:
        raise ValueError("duplicate query_id")
    if len({(j.query_id, j.item_id) for j in judgments}) != 12:
        raise ValueError("duplicate judgment pair")

    # content groups
    groups: dict[str, list[PilotCanonicalDocument]] = {}
    for d in documents:
        groups.setdefault(d.content_group_id, []).append(d)
    if len(groups) != 6:
        raise ValueError("expected 6 content groups")
    for group, docs in groups.items():
        if len(docs) != 2:
            raise ValueError(f"{group}: expected exactly 2 documents")
        texts = {d.text_normalized for d in docs}
        hashes = {d.source_text_sha256 for d in docs}
        if len(texts) != 1:
            raise ValueError(f"{group}: documents must share normalized text")
        if len(hashes) != 1:
            raise ValueError(f"{group}: documents must share source_text_sha256")
        if len({d.doc_id for d in docs}) != 2:
            raise ValueError(f"{group}: doc_ids must be distinct")

    # chunks
    chunks_by_doc: dict[str, list[PilotCanonicalChunk]] = {}
    for c in chunks:
        chunks_by_doc.setdefault(c.doc_id, []).append(c)
    for doc in documents:
        cs = sorted(chunks_by_doc.get(doc.doc_id, []), key=lambda c: c.source_order)
        if [c.source_order for c in cs] != [0, 1]:
            raise ValueError(f"{doc.doc_id}: chunk orders must be [0, 1]")
        reconstructed = _normalize(cs[0].text + " " + cs[1].text)
        if reconstructed != doc.text_normalized:
            raise ValueError(f"{doc.doc_id}: chunks do not reconstruct document")
        for c in cs:
            if c.content_group_id != doc.content_group_id:
                raise ValueError(f"{c.chunk_id}: content group mismatch")

    # queries
    group_to_query: dict[str, PilotCanonicalQuery] = {}
    for q in queries:
        if q.target_content_group_id in group_to_query:
            raise ValueError("multiple queries for the same content group")
        if q.target_content_group_id not in groups:
            raise ValueError(f"{q.query_id}: unknown target content group")
        group_to_query[q.target_content_group_id] = q
    if set(group_to_query) != set(groups):
        raise ValueError("every content group must have exactly one query")

    # judgments
    j_by_query: dict[str, list[PilotCanonicalJudgment]] = {}
    for j in judgments:
        j_by_query.setdefault(j.query_id, []).append(j)
    doc_by_id = {d.doc_id: d for d in documents}
    q_by_id = {q.query_id: q for q in queries}
    for qid, js in j_by_query.items():
        if len(js) != 2:
            raise ValueError(f"{qid}: expected exactly 2 judgments")
        q = q_by_id[qid]
        for j in js:
            if j.item_id not in doc_by_id:
                raise ValueError(f"{qid}: unknown item {j.item_id}")
            if doc_by_id[j.item_id].content_group_id != q.target_content_group_id:
                raise ValueError(f"{qid}: judgment outside target content group")
