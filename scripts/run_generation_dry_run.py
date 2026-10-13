"""Run the provider-neutral generation pipeline without model inference."""

from __future__ import annotations

from pathlib import Path

from rag_system.artifacts import (
    ArtifactComponentReference,
    build_file_sidecar_manifest,
    get_git_revision,
    read_json_model,
    sha256_fingerprint,
    write_json_model_atomic,
)
from rag_system.generation import (
    DeterministicDryRunAdapter,
    GenerationPrompt,
    GenerationRequest,
    build_generated_answer,
)

PROMPT_PATH = Path(
    "outputs/clerk_generation_prompt_smoke.json"
)

ANSWER_PATH = Path(
    "outputs/clerk_generation_runtime_dry_run.json"
)

MANIFEST_PATH = Path(
    "outputs/clerk_generation_runtime_dry_run.manifest.json"
)

MODEL_ID = "synthetic-runtime-contract-model"
MAXIMUM_OUTPUT_TOKENS = 1_024
TEMPERATURE = 0.0
SEED = 20_261_003


def main() -> None:
    """Execute the complete normalized runtime pipeline."""

    prompt = read_json_model(
        PROMPT_PATH,
        GenerationPrompt,
    )

    if len(prompt.citation_ids) < 2:
        raise RuntimeError(
            "The dry-run prompt requires at least two citations."
        )

    answer_text = (
        "The retrieved materials describe the relevant legal "
        f"and technical background {prompt.citation_ids[0]}. "
        "A separate retrieved source provides additional "
        f"supporting context {prompt.citation_ids[1]}."
    )

    request = GenerationRequest(
        prompt=prompt,
        model_id=MODEL_ID,
        maximum_output_tokens=(
            MAXIMUM_OUTPUT_TOKENS
        ),
        temperature=TEMPERATURE,
        seed=SEED,
        response_format="text",
    )

    adapter = DeterministicDryRunAdapter(
        answer_text=answer_text,
    )

    response = adapter.generate(
        request
    )

    answer = build_generated_answer(
        request=request,
        response=response,
        metadata={
            "artifact_validation_only": True,
            "production_answer": False,
            "scientific_result": False,
            "hosted_endpoint_configured": False,
            "runtime_contract_validated": True,
        },
    )

    if not answer.citation_validation.valid:
        raise RuntimeError(
            "The dry-run answer failed citation validation."
        )

    if (
        response.metadata is None
        or response.metadata.get(
            "model_inference_performed"
        )
        is not False
    ):
        raise RuntimeError(
            "Dry-run response incorrectly indicates inference."
        )

    write_json_model_atomic(
        ANSWER_PATH,
        answer,
        overwrite=True,
    )

    configuration = {
        "runtime_id": adapter.runtime_id,
        "model_id": MODEL_ID,
        "maximum_output_tokens": (
            MAXIMUM_OUTPUT_TOKENS
        ),
        "temperature": TEMPERATURE,
        "seed": SEED,
        "response_format": "text",
        "model_inference_performed": False,
    }

    component = ArtifactComponentReference(
        component_type="generation_runtime_adapter",
        component_id=adapter.runtime_id,
        component_version="1.0.0",
        configuration_hash=(
            sha256_fingerprint(
                configuration
            )
        ),
        configuration=configuration,
    )

    sidecar = build_file_sidecar_manifest(
        artifact_type="generated_answer",
        artifact_path=ANSWER_PATH,
        record_count=1,
        component=component,
        dataset_id=prompt.dataset_id,
        code_revision=get_git_revision(),
        execution_profile="dry_run",
        metadata={
            "answer_id": answer.answer_id,
            "prompt_id": answer.prompt_id,
            "query_id": answer.query_id,
            "runtime_id": adapter.runtime_id,
            "runtime_contract_validated": True,
            "model_inference_performed": False,
            "hosted_endpoint_configured": False,
            "production_answer": False,
            "citation_validation_valid": (
                answer.citation_validation.valid
            ),
        },
    )

    write_json_model_atomic(
        MANIFEST_PATH,
        sidecar,
        overwrite=True,
    )

    print("[OK] Provider-neutral generation dry run")
    print("  Runtime:", adapter.runtime_id)
    print("  Model ID:", answer.generation_model_id)
    print("  Answer ID:", answer.answer_id)
    print("  Prompt ID:", answer.prompt_id)
    print(
        "  Input tokens:",
        answer.input_token_count,
    )
    print(
        "  Output tokens:",
        answer.output_token_count,
    )
    print(
        "  Used citations:",
        answer.citation_validation.used_citation_ids,
    )
    print(
        "  Unknown citations:",
        answer.citation_validation.unknown_citation_ids,
    )
    print(
        "  Citation validation:",
        answer.citation_validation.valid,
    )
    print(
        "  Model inference:",
        response.metadata[
            "model_inference_performed"
        ],
    )
    print("  Answer:", ANSWER_PATH.resolve())
    print("  Manifest:", MANIFEST_PATH.resolve())


if __name__ == "__main__":
    main()
