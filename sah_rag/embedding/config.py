"""Embedding-generation configuration."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from sah_rag.embedding.models import (
    EmbeddingKind,
    VectorPrecision,
)


class EmbeddingConfig(BaseModel):
    """Configuration shared by embedding generation runs."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    embedder_id: str = "bge_m3_embedder"
    embedder_version: str = "0.1.0"

    model_id: str = "BAAI/bge-m3"
    model_revision: str

    tokenizer_id: str

    embedding_kind: EmbeddingKind = EmbeddingKind.HYBRID

    dense_dimension: int | None = Field(
        default=1024,
        gt=0,
    )

    dense_precision: VectorPrecision | None = VectorPrecision.FLOAT32

    dense_normalized: bool | None = True

    device: str = "cpu"
    use_fp16: bool = False

    batch_size: int = Field(
        default=2,
        gt=0,
    )

    maximum_model_input_tokens: int = Field(
        default=8192,
        gt=0,
    )

    shard_maximum_records: int = Field(
        default=10000,
        gt=0,
    )

    @model_validator(mode="after")
    def validate_embedding_config(
        self,
    ) -> EmbeddingConfig:
        """Validate embedding representation configuration."""

        required_values = (
            self.embedder_id,
            self.embedder_version,
            self.model_id,
            self.model_revision,
            self.tokenizer_id,
            self.device,
        )

        if any(not value.strip() for value in required_values):
            raise ValueError("Embedding configuration fields cannot be empty.")

        dense_required = self.embedding_kind in {
            EmbeddingKind.DENSE,
            EmbeddingKind.HYBRID,
        }

        if dense_required and any(
            value is None
            for value in (
                self.dense_dimension,
                self.dense_precision,
                self.dense_normalized,
            )
        ):
            raise ValueError("Dense and hybrid embedding configurations require dense metadata.")

        if self.use_fp16 and self.device == "cpu":
            raise ValueError("FP16 execution is not enabled for the initial CPU profile.")

        return self
