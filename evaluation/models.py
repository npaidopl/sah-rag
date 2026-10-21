"""Typed artifacts for retrieval evaluation."""

from __future__ import annotations

from pydantic import Field, model_validator

from rag_system.artifacts.models import ArtifactBaseModel


class RetrievalJudgment(ArtifactBaseModel):
    """Relevance judgment for one retrievable identifier."""

    item_id: str
    relevance: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_item_id(self) -> RetrievalJudgment:
        """Require a non-empty judged identifier."""
        if not self.item_id.strip():
            raise ValueError(
                "Judged item_id cannot be empty."
            )

        return self


class RetrievalEvaluationCase(ArtifactBaseModel):
    """One query, judgments, and ranked retrieval output."""

    query_id: str
    query_text: str
    retrieval_method: str
    evaluation_level: str

    judgments: tuple[RetrievalJudgment, ...]
    retrieved_ids: tuple[str, ...]

    metadata: dict[str, object] = Field(
        default_factory=dict
    )

    @model_validator(mode="after")
    def validate_case(self) -> RetrievalEvaluationCase:
        """Validate query identity and unique data."""
        required_values = (
            self.query_id,
            self.query_text,
            self.retrieval_method,
            self.evaluation_level,
        )

        if any(
            not value.strip()
            for value in required_values
        ):
            raise ValueError(
                "Evaluation identity fields cannot be empty."
            )

        judged_ids = [
            judgment.item_id
            for judgment in self.judgments
        ]

        if len(judged_ids) != len(set(judged_ids)):
            raise ValueError(
                "Judgments contain duplicate item IDs."
            )

        if (
            len(self.retrieved_ids)
            != len(set(self.retrieved_ids))
        ):
            raise ValueError(
                "Retrieved IDs must be unique."
            )

        return self


class RetrievalMetrics(ArtifactBaseModel):
    """Metrics for one evaluated retrieval case."""

    query_id: str
    retrieval_method: str
    evaluation_level: str
    cutoff: int = Field(gt=0)

    relevant_count: int = Field(ge=0)
    retrieved_count: int = Field(ge=0)
    relevant_retrieved_count: int = Field(ge=0)

    recall_at_k: float = Field(ge=0.0, le=1.0)
    precision_at_k: float = Field(ge=0.0, le=1.0)
    reciprocal_rank: float = Field(ge=0.0, le=1.0)
    ndcg_at_k: float = Field(ge=0.0, le=1.0)
    hit_at_k: float = Field(ge=0.0, le=1.0)


class RetrievalAggregateMetrics(ArtifactBaseModel):
    """Macro-averaged metrics over multiple queries."""

    retrieval_method: str
    evaluation_level: str
    cutoff: int = Field(gt=0)
    query_count: int = Field(gt=0)

    mean_recall_at_k: float = Field(ge=0.0, le=1.0)
    mean_precision_at_k: float = Field(ge=0.0, le=1.0)
    mean_reciprocal_rank: float = Field(ge=0.0, le=1.0)
    mean_ndcg_at_k: float = Field(ge=0.0, le=1.0)
    mean_hit_at_k: float = Field(ge=0.0, le=1.0)


class RetrievalEvaluationRun(ArtifactBaseModel):
    """Summary of one multi-method retrieval evaluation run."""

    artifact_type: str = "retrieval_evaluation_run"
    artifact_version: str = "1.0"

    run_id: str
    dataset_id: str
    evaluation_level: str

    retrieval_methods: tuple[str, ...]
    cutoffs: tuple[int, ...]

    case_count: int = Field(gt=0)
    metric_record_count: int = Field(gt=0)
    aggregate_record_count: int = Field(gt=0)

    cases_path: str
    metrics_path: str
    aggregates_path: str

    metadata: dict[str, object] = Field(
        default_factory=dict
    )

    @model_validator(mode="after")
    def validate_run(self) -> RetrievalEvaluationRun:
        """Validate evaluation-run identity and dimensions."""
        required_values = (
            self.run_id,
            self.dataset_id,
            self.evaluation_level,
            self.cases_path,
            self.metrics_path,
            self.aggregates_path,
        )

        if any(
            not value.strip()
            for value in required_values
        ):
            raise ValueError(
                "Evaluation-run identity fields cannot be empty."
            )

        if not self.retrieval_methods:
            raise ValueError(
                "At least one retrieval method is required."
            )

        if len(self.retrieval_methods) != len(
            set(self.retrieval_methods)
        ):
            raise ValueError(
                "Retrieval methods must be unique."
            )

        if not self.cutoffs:
            raise ValueError(
                "At least one cutoff is required."
            )

        if any(cutoff < 1 for cutoff in self.cutoffs):
            raise ValueError(
                "Evaluation cutoffs must be positive."
            )

        if tuple(sorted(set(self.cutoffs))) != self.cutoffs:
            raise ValueError(
                "Evaluation cutoffs must be unique and sorted."
            )

        return self
