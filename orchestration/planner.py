"""Common orchestrator skeleton and dry-run planner (Step 68, through 68D).

This module builds a typed, non-executing plan for one experiment cell.
It does NOT execute any architecture, index, embedding, LLM, retrieval,
evaluation, foreign code, or dataset transformation. It does NOT persist
anything to disk; persistence (Step 68E), validation scripts (68F),
tests (68G), and smoke (68I) are pending the full Step 68 specification.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import yaml  # type: ignore[import-untyped]
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from rag_system.artifacts import sha256_fingerprint
from rag_system.execution_contracts import (
    ArchitectureExecutionManifest,
    ExecutionResourceUsage,
    ExperimentCellIdentity,
    build_experiment_cell_id,
)

ORCHESTRATOR_CONTRACT_VERSION = "0.1.0"
EXECUTION_CONTRACT_VERSION = "0.1.0"
EVALUATION_CONTRACT_VERSION = "0.1.0"

WAVE1_ARCHITECTURES: tuple[str, ...] = (
    "longrag",
    "raptor",
    "hipporag2",
    "lightrag",
    "pathrag",
)

PATHRAG_REQUIRED_LABEL = "PathRAG paper-derived architectural comparator"

WAVE1_CLASSIFICATION: dict[str, str] = {
    "longrag": "CONTROLLED_ADAPTATION",
    "raptor": "CONTROLLED_ADAPTATION",
    "hipporag2": "FAITHFUL",
    "lightrag": "CONTROLLED_ADAPTATION",
    "pathrag": "ARCHITECTURAL_COMPARATOR",
}

WAVE1_IMPLEMENTATION_CLASS: dict[str, str] = {
    "longrag": "CONTROLLED_ADAPTATION",
    "raptor": "CONTROLLED_ADAPTATION",
    "hipporag2": "FAITHFUL",
    "lightrag": "CONTROLLED_ADAPTATION",
    "pathrag": "PAPER_DERIVED_REIMPLEMENTATION",
}

READINESS_REGISTRY_PATH = Path("baseline_audit/wave1_readiness_registry.yaml")

STAGE_SEQUENCE: tuple[tuple[str, int, tuple[str, ...]], ...] = (
    ("preflight", 1, ()),
    ("input_snapshot", 2, ("preflight",)),
    ("architecture_preprocessing", 3, ("input_snapshot",)),
    ("architecture_indexing", 4, ("architecture_preprocessing",)),
    ("architecture_retrieval", 5, ("architecture_indexing",)),
    ("canonical_document_expansion", 6, ("architecture_retrieval",)),
    ("common_evaluation", 7, ("canonical_document_expansion",)),
    ("artifact_finalization", 8, ("common_evaluation",)),
)

VALID_STAGE_STATUS = ("PLANNED", "BLOCKED", "SKIPPED")


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class OrchestratorExecutionRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    dataset_id: str
    dataset_revision: str
    split: str

    architecture_id: str
    adapter_id: str
    adapter_version: str
    architecture_configuration_hash: str
    canonical_input_manifest_sha256: str

    random_seed: int = Field(ge=0)
    project_code_revision: str
    execution_environment_id: str

    dataset_root: str
    output_root: str

    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator(
        "dataset_id",
        "dataset_revision",
        "split",
        "architecture_id",
        "adapter_id",
        "adapter_version",
        "architecture_configuration_hash",
        "canonical_input_manifest_sha256",
        "project_code_revision",
        "execution_environment_id",
        "dataset_root",
        "output_root",
    )
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v:
            raise ValueError("must be non-empty")
        return v

    @field_validator("architecture_id")
    @classmethod
    def _registered(cls, v: str) -> str:
        if v not in WAVE1_ARCHITECTURES:
            raise ValueError(f"architecture not in Wave 1: {v}")
        return v

    @model_validator(mode="after")
    def _pathrag_guard(self) -> OrchestratorExecutionRequest:
        if (
            self.architecture_id == "pathrag"
            and self.metadata.get("foreign_implementation")
        ):
            raise ValueError(
                "PathRAG request must not declare a foreign implementation"
            )
        return self


class OrchestratorStagePlan(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    stage_id: str
    stage_order: int = Field(gt=0)
    stage_status: str
    architecture_id: str

    input_paths: tuple[str, ...] = ()
    planned_output_paths: tuple[str, ...] = ()
    depends_on: tuple[str, ...] = ()

    execution_enabled: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("stage_status")
    @classmethod
    def _valid_status(cls, v: str) -> str:
        if v not in VALID_STAGE_STATUS:
            raise ValueError(f"invalid stage_status: {v}")
        return v

    @field_validator("execution_enabled")
    @classmethod
    def _no_execution(cls, v: bool) -> bool:
        if v is not False:
            raise ValueError("stage execution_enabled must be False")
        return v

    @field_validator("input_paths", "planned_output_paths", "depends_on")
    @classmethod
    def _unique(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(v)) != len(v):
            raise ValueError("paths and dependencies must be unique")
        return v


class OrchestratorArtifactLayout(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    experiment_root: str

    input_directory: str
    architecture_directory: str
    retrieval_directory: str
    evaluation_directory: str
    logs_directory: str
    checkpoints_directory: str

    dry_run_plan_path: str
    execution_manifest_path: str
    plan_manifest_path: str
    checksums_path: str

    @model_validator(mode="after")
    def _under_root(self) -> OrchestratorArtifactLayout:
        root = self.experiment_root.rstrip("/")
        children = (
            self.input_directory,
            self.architecture_directory,
            self.retrieval_directory,
            self.evaluation_directory,
            self.logs_directory,
            self.checkpoints_directory,
            self.dry_run_plan_path,
            self.execution_manifest_path,
            self.plan_manifest_path,
            self.checksums_path,
        )
        for c in children:
            if not c.startswith(root + "/"):
                raise ValueError(f"child path outside experiment_root: {c}")
        return self


class OrchestratorDryRunPlan(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    plan_id: str
    experiment_cell: ExperimentCellIdentity
    architecture_display_name: str
    architecture_classification: str
    implementation_class: str

    request: OrchestratorExecutionRequest
    artifact_layout: OrchestratorArtifactLayout
    stages: tuple[OrchestratorStagePlan, ...]

    readiness_status: str
    readiness_declaration: str

    execution_contract_version: str
    evaluation_contract_version: str
    orchestrator_contract_version: str

    architecture_execution_enabled: bool = False
    evaluation_execution_enabled: bool = False
    final_generation_enabled: bool = False

    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator(
        "architecture_execution_enabled",
        "evaluation_execution_enabled",
        "final_generation_enabled",
    )
    @classmethod
    def _flags_false(cls, v: bool) -> bool:
        if v is not False:
            raise ValueError("execution flags must all be False")
        return v


class OrchestratorPlanManifest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    manifest_id: str
    plan_id: str
    experiment_cell_id: str
    dataset_id: str
    dataset_revision: str
    split: str
    architecture_id: str

    request_sha256: str
    plan_sha256: str

    orchestrator_contract_version: str = ORCHESTRATOR_CONTRACT_VERSION
    execution_contract_version: str = EXECUTION_CONTRACT_VERSION
    evaluation_contract_version: str = EVALUATION_CONTRACT_VERSION

    architecture_execution_enabled: bool = False
    evaluation_execution_enabled: bool = False
    final_generation_enabled: bool = False

    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator(
        "architecture_execution_enabled",
        "evaluation_execution_enabled",
        "final_generation_enabled",
    )
    @classmethod
    def _flags_false(cls, v: bool) -> bool:
        if v is not False:
            raise ValueError("manifest execution flags must all be False")
        return v


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_wave1_readiness_registry() -> dict[str, Any]:
    if not READINESS_REGISTRY_PATH.is_file():
        raise FileNotFoundError(
            f"readiness registry not found: {READINESS_REGISTRY_PATH}"
        )
    data = yaml.safe_load(READINESS_REGISTRY_PATH.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(
            f"readiness registry must be a mapping: {READINESS_REGISTRY_PATH}"
        )
    return cast("dict[str, Any]", data)


def _sha256_json(payload: Any) -> str:
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _plan_id(
    *,
    experiment_cell_id: str,
    execution_environment_id: str,
    orchestrator_contract_version: str,
) -> str:
    fp = sha256_fingerprint(
        {
            "kind": "orchestrator_dry_run_plan",
            "experiment_cell_id": experiment_cell_id,
            "execution_environment_id": execution_environment_id,
            "orchestrator_contract_version": orchestrator_contract_version,
        }
    )
    return f"orchestrator-plan-{fp}"


# ---------------------------------------------------------------------------
# Planners (68C)
# ---------------------------------------------------------------------------


def build_orchestrator_artifact_layout(
    *,
    request: OrchestratorExecutionRequest,
    experiment_cell_id: str,
) -> OrchestratorArtifactLayout:
    root = (
        f"{request.output_root.rstrip('/')}/"
        f"{request.dataset_id}/"
        f"{request.architecture_id}/"
        f"{experiment_cell_id}"
    )
    return OrchestratorArtifactLayout(
        experiment_root=root,
        input_directory=f"{root}/input",
        architecture_directory=f"{root}/architecture",
        retrieval_directory=f"{root}/retrieval",
        evaluation_directory=f"{root}/evaluation",
        logs_directory=f"{root}/logs",
        checkpoints_directory=f"{root}/checkpoints",
        dry_run_plan_path=f"{root}/dry-run.plan.json",
        execution_manifest_path=f"{root}/execution.manifest.json",
        plan_manifest_path=f"{root}/dry-run.manifest.json",
        checksums_path=f"{root}/checksums.json",
    )


def build_orchestrator_stage_plan(
    *,
    architecture_id: str,
    artifact_layout: OrchestratorArtifactLayout,
) -> tuple[OrchestratorStagePlan, ...]:
    stages: list[OrchestratorStagePlan] = []
    for stage_id, order, deps in STAGE_SEQUENCE:
        stages.append(
            OrchestratorStagePlan(
                stage_id=stage_id,
                stage_order=order,
                stage_status="PLANNED",
                architecture_id=architecture_id,
                input_paths=(),
                planned_output_paths=(),
                depends_on=deps,
                execution_enabled=False,
                metadata={"artifact_root": artifact_layout.experiment_root},
            )
        )
    return tuple(stages)


def build_orchestrator_dry_run_plan(
    *,
    request: OrchestratorExecutionRequest,
    readiness_registry: dict[str, Any],
) -> OrchestratorDryRunPlan:
    architectures = readiness_registry.get("architectures", {})
    architecture_id = request.architecture_id
    if architecture_id not in architectures:
        raise ValueError(
            f"architecture not in Wave-1 readiness registry: {architecture_id}"
        )
    entry = architectures[architecture_id]

    if entry["readiness_status"] != "READY_FOR_DATA":
        raise ValueError(
            f"architecture {architecture_id} is not READY_FOR_DATA"
        )
    if entry.get("final_generation_enabled") is not False:
        raise ValueError("readiness entry final_generation_enabled must be False")

    cell_id = build_experiment_cell_id(
        dataset_id=request.dataset_id,
        dataset_revision=request.dataset_revision,
        split=request.split,
        architecture_id=request.architecture_id,
        adapter_id=request.adapter_id,
        adapter_version=request.adapter_version,
        architecture_configuration_hash=request.architecture_configuration_hash,
        canonical_input_manifest_sha256=request.canonical_input_manifest_sha256,
        random_seed=request.random_seed,
        code_revision=request.project_code_revision,
    )

    identity = ExperimentCellIdentity(
        experiment_cell_id=cell_id,
        dataset_id=request.dataset_id,
        dataset_revision=request.dataset_revision,
        split=request.split,
        architecture_id=request.architecture_id,
        adapter_id=request.adapter_id,
        adapter_version=request.adapter_version,
        architecture_configuration_hash=request.architecture_configuration_hash,
        canonical_input_manifest_sha256=request.canonical_input_manifest_sha256,
        random_seed=request.random_seed,
        code_revision=request.project_code_revision,
    )

    layout = build_orchestrator_artifact_layout(
        request=request, experiment_cell_id=cell_id
    )
    stages = build_orchestrator_stage_plan(
        architecture_id=architecture_id, artifact_layout=layout
    )

    display_name = entry["execution_label"]
    classification = WAVE1_CLASSIFICATION[architecture_id]
    implementation_class = WAVE1_IMPLEMENTATION_CLASS[architecture_id]

    if architecture_id == "pathrag":
        if display_name != PATHRAG_REQUIRED_LABEL:
            raise ValueError(
                "PathRAG execution_label must be the full comparator label"
            )
        if entry.get("foreign_repository_status") != "BLOCKED":
            raise ValueError(
                "PathRAG foreign_repository_status must remain BLOCKED"
            )
        if implementation_class != "PAPER_DERIVED_REIMPLEMENTATION":
            raise ValueError(
                "PathRAG implementation_class must be PAPER_DERIVED_REIMPLEMENTATION"
            )

    plan = OrchestratorDryRunPlan(
        plan_id=_plan_id(
            experiment_cell_id=cell_id,
            execution_environment_id=request.execution_environment_id,
            orchestrator_contract_version=ORCHESTRATOR_CONTRACT_VERSION,
        ),
        experiment_cell=identity,
        architecture_display_name=display_name,
        architecture_classification=classification,
        implementation_class=implementation_class,
        request=request,
        artifact_layout=layout,
        stages=stages,
        readiness_status=entry["readiness_status"],
        readiness_declaration=entry["readiness_declaration"],
        execution_contract_version=EXECUTION_CONTRACT_VERSION,
        evaluation_contract_version=EVALUATION_CONTRACT_VERSION,
        orchestrator_contract_version=ORCHESTRATOR_CONTRACT_VERSION,
        architecture_execution_enabled=False,
        evaluation_execution_enabled=False,
        final_generation_enabled=False,
        metadata={},
    )
    validate_orchestrator_dry_run_plan(plan)
    return plan


def validate_orchestrator_dry_run_plan(plan: OrchestratorDryRunPlan) -> None:
    expected_cell_id = build_experiment_cell_id(
        dataset_id=plan.request.dataset_id,
        dataset_revision=plan.request.dataset_revision,
        split=plan.request.split,
        architecture_id=plan.request.architecture_id,
        adapter_id=plan.request.adapter_id,
        adapter_version=plan.request.adapter_version,
        architecture_configuration_hash=plan.request.architecture_configuration_hash,
        canonical_input_manifest_sha256=plan.request.canonical_input_manifest_sha256,
        random_seed=plan.request.random_seed,
        code_revision=plan.request.project_code_revision,
    )
    if expected_cell_id != plan.experiment_cell.experiment_cell_id:
        raise ValueError("experiment_cell does not match request")

    if plan.readiness_status != "READY_FOR_DATA":
        raise ValueError("readiness_status must be READY_FOR_DATA")

    if len(plan.stages) != 8:
        raise ValueError("exactly eight stages required")

    orders = sorted(s.stage_order for s in plan.stages)
    if orders != list(range(1, 9)):
        raise ValueError("stage orders must be 1..8")

    stage_ids = {s.stage_id for s in plan.stages}
    for s in plan.stages:
        for dep in s.depends_on:
            if dep not in stage_ids:
                raise ValueError(f"unknown dependency {dep} for stage {s.stage_id}")

    for flag in (
        plan.architecture_execution_enabled,
        plan.evaluation_execution_enabled,
        plan.final_generation_enabled,
    ):
        if flag is not False:
            raise ValueError("execution flags must all be False")

    if plan.request.architecture_id == "pathrag":
        if plan.architecture_display_name != PATHRAG_REQUIRED_LABEL:
            raise ValueError("PathRAG label mismatch")
        if plan.implementation_class != "PAPER_DERIVED_REIMPLEMENTATION":
            raise ValueError("PathRAG implementation_class mismatch")


# ---------------------------------------------------------------------------
# 68D — planned execution manifest
# ---------------------------------------------------------------------------


def build_planned_execution_manifest(
    *,
    plan: OrchestratorDryRunPlan,
) -> ArchitectureExecutionManifest:
    """Produce the PLANNED ArchitectureExecutionManifest for a dry run.

    Because the frozen ArchitectureExecutionManifest requires
    ``started_at_utc``, we use the Unix epoch as a placeholder and mark
    the manifest via metadata as dry-run-only.
    """
    epoch = datetime(1970, 1, 1, tzinfo=UTC)

    return ArchitectureExecutionManifest(
        experiment_cell_id=plan.experiment_cell.experiment_cell_id,
        architecture_id=plan.request.architecture_id,
        architecture_display_name=plan.architecture_display_name,
        architecture_classification=plan.architecture_classification,
        implementation_class=plan.implementation_class,
        adapter_id=plan.request.adapter_id,
        adapter_version=plan.request.adapter_version,
        architecture_source_revision=plan.request.project_code_revision,
        project_code_revision=plan.request.project_code_revision,
        dataset_id=plan.request.dataset_id,
        dataset_revision=plan.request.dataset_revision,
        split=plan.request.split,
        random_seed=plan.request.random_seed,
        configuration_hash=plan.request.architecture_configuration_hash,
        canonical_input_manifest_sha256=(
            plan.request.canonical_input_manifest_sha256
        ),
        execution_environment_id=plan.request.execution_environment_id,
        execution_status="PLANNED",
        started_at_utc=epoch,
        completed_at_utc=None,
        failure=None,
        resource_usage=ExecutionResourceUsage(),
        artifact_inventory=(),
        safety={
            "final_generation_enabled": False,
            "answer_evaluation_enabled": False,
            "dry_run_only": True,
            "architecture_execution_performed": False,
            "evaluation_execution_performed": False,
        },
        metadata={
            "plan_id": plan.plan_id,
            "dry_run_only": True,
            "orchestrator_contract_version": plan.orchestrator_contract_version,
        },
    )


def build_orchestrator_plan_manifest(
    *,
    plan: OrchestratorDryRunPlan,
) -> OrchestratorPlanManifest:
    request_sha = _sha256_json(plan.request.model_dump(mode="json"))
    plan_sha = _sha256_json(plan.model_dump(mode="json"))
    fp = sha256_fingerprint(
        {
            "kind": "orchestrator_plan_manifest",
            "plan_id": plan.plan_id,
            "request_sha256": request_sha,
        }
    )
    return OrchestratorPlanManifest(
        manifest_id=f"orchestrator-manifest-{fp}",
        plan_id=plan.plan_id,
        experiment_cell_id=plan.experiment_cell.experiment_cell_id,
        dataset_id=plan.request.dataset_id,
        dataset_revision=plan.request.dataset_revision,
        split=plan.request.split,
        architecture_id=plan.request.architecture_id,
        request_sha256=request_sha,
        plan_sha256=plan_sha,
        orchestrator_contract_version=plan.orchestrator_contract_version,
        execution_contract_version=plan.execution_contract_version,
        evaluation_contract_version=plan.evaluation_contract_version,
        architecture_execution_enabled=False,
        evaluation_execution_enabled=False,
        final_generation_enabled=False,
        metadata={},
    )
