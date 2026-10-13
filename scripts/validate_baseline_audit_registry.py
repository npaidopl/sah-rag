"""Validate the Baseline Audit Registry."""

from __future__ import annotations

from pathlib import Path

import yaml

REGISTRY_PATH = Path("baseline_audit/registry.yaml")

EXPECTED_ARCHITECTURES = {
    "disretrieval",
    "hikey",
    "raptor",
    "longrag",
    "memorag",
    "kag",
    "metadata_rag",
    "hipporag2",
    "lightrag",
    "pathrag",
}

VALID_WAVES = {
    1,
    2,
    3,
}

VALID_INITIAL_STATUS_PREFIXES = (
    "GO",
    "HOLD",
    "BLOCKED",
)


def main() -> None:
    """Validate registry structure and baseline identities."""
    data = yaml.safe_load(REGISTRY_PATH.read_text(encoding="utf-8"))

    registry = data["registry"]
    architectures = data["architectures"]

    if registry["architecture_count"] != 10:
        raise RuntimeError("Registry architecture count must be ten.")

    architecture_ids = {architecture["architecture_id"] for architecture in architectures}

    if architecture_ids != EXPECTED_ARCHITECTURES:
        raise RuntimeError("Registry architecture IDs are incomplete.")

    if len(architectures) != len(architecture_ids):
        raise RuntimeError("Registry contains duplicate architecture IDs.")

    for architecture in architectures:
        architecture_id = architecture["architecture_id"]

        paper = architecture["paper"]
        repository = architecture["repository"]
        audit = architecture["audit"]

        if not paper["title"].strip():
            raise RuntimeError(f"{architecture_id}: missing paper title.")

        if not paper["arxiv_id"].strip():
            raise RuntimeError(f"{architecture_id}: missing arXiv ID.")

        if audit["wave"] not in VALID_WAVES:
            raise RuntimeError(f"{architecture_id}: invalid audit wave.")

        if not str(audit["initial_status"]).startswith(VALID_INITIAL_STATUS_PREFIXES):
            raise RuntimeError(f"{architecture_id}: invalid initial status.")

        if repository["official"] is True:
            repository_url = repository.get("url")

            if not repository_url:
                raise RuntimeError(f"{architecture_id}: official repository has no URL.")

            if not repository_url.startswith("https://github.com/"):
                raise RuntimeError(f"{architecture_id}: repository is not a GitHub HTTPS URL.")

        if repository["resolved_commit"] is not None:
            commit = str(repository["resolved_commit"])

            if len(commit) != 40:
                raise RuntimeError(f"{architecture_id}: resolved commit must be a full SHA.")

    pathrag_record = None
    for arch in architectures:
        if arch.get("architecture_id") == "pathrag":
            pathrag_record = arch
            break
    if pathrag_record is None:
        raise RuntimeError("PathRAG record missing.")
    pa = pathrag_record["audit"]
    if pa["adaptation_class"] != "ARCHITECTURAL_COMPARATOR":
        raise RuntimeError(
            "PathRAG must remain an architectural comparator."
        )
    if (
        pa["implementation_class"]
        != "PAPER_DERIVED_REIMPLEMENTATION"
    ):
        raise RuntimeError(
            "PathRAG requires a paper-derived implementation."
        )
    if pa["foreign_execution_decision"] != "BLOCKED":
        raise RuntimeError(
            "Foreign PathRAG execution must remain blocked."
        )
    if (
        pa["project_owned_specification_decision"]
        != "CONDITIONAL_GO"
    ):
        raise RuntimeError(
            "PathRAG clean-room plan must remain CONDITIONAL_GO."
        )
    if pa["foreign_repository_status"] != "BLOCKED":
        raise RuntimeError(
            "PathRAG foreign repository status must remain BLOCKED."
        )
    comparator = pa["comparator"]
    if (
        comparator["implementation_class"]
        != "PAPER_DERIVED_REIMPLEMENTATION"
    ):
        raise RuntimeError(
            "PathRAG comparator must remain paper-derived."
        )
    if comparator["readiness_status"] != "READY_FOR_DATA":
        raise RuntimeError(
            "PathRAG comparator must be READY_FOR_DATA."
        )
    if comparator["synthetic_end_to_end_smoke"] != "passed":
        raise RuntimeError(
            "PathRAG comparator smoke must remain passed."
        )
    if (
        comparator["readiness_declaration"]
        != "baseline_audit/pathrag_readiness.yaml"
    ):
        raise RuntimeError(
            "Unexpected PathRAG readiness declaration."
        )

    print("[OK] Baseline Audit Registry")
    print(
        " Registry version:",
        registry["version"],
    )
    print(
        " Frozen reference:",
        registry["created_for"],
    )
    print(
        " Architecture count:",
        len(architectures),
    )

    for wave in sorted(VALID_WAVES):
        wave_architectures = [
            architecture["architecture_id"]
            for architecture in architectures
            if architecture["audit"]["wave"] == wave
        ]

        print(
            f" Wave {wave}:",
            ", ".join(wave_architectures),
        )


if __name__ == "__main__":
    main()
