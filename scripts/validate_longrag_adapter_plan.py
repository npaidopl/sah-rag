"""Validate the LongRAG environment and adapter specifications."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ENVIRONMENT_PATH = Path(
    "baseline_audit/"
    "longrag_environment_plan.yaml"
)

ADAPTER_PATH = Path(
    "baseline_audit/"
    "longrag_adapter_spec.yaml"
)

EXPECTED_COMMIT = "ba92f95"

REQUIRED_UNIT_FIELDS = {
    "long_unit_id",
    "dataset_id",
    "construction_policy",
    "unit_order",
    "text",
    "token_count",
    "member_doc_ids",
    "member_parent_ids",
    "member_chunk_ids",
    "metadata",
}

REQUIRED_RESULT_FIELDS = {
    "rank",
    "long_unit_id",
    "score",
    "member_doc_ids",
    "construction_policy",
}

REQUIRED_METRICS = {
    "recall_at_k",
    "precision_at_k",
    "reciprocal_rank",
    "ndcg_at_k",
    "hit_at_k",
}


def load_yaml(path: Path) -> dict[str, Any]:
    """Load one YAML mapping."""
    value = yaml.safe_load(
        path.read_text(
            encoding="utf-8"
        )
    )
    if not isinstance(value, dict):
        raise RuntimeError(
            f"Expected YAML mapping: {path}"
        )
    return value


def main() -> None:
    """Validate the complete LongRAG planning contract."""
    environment = load_yaml(
        ENVIRONMENT_PATH
    )
    adapter = load_yaml(
        ADAPTER_PATH
    )

    architecture = environment[
        "architecture"
    ]
    if architecture["id"] != "longrag":
        raise RuntimeError(
            "Environment plan must target LongRAG."
        )

    if architecture[
        "execution_decision"
    ] != "CONDITIONAL_GO":
        raise RuntimeError(
            "LongRAG decision must remain CONDITIONAL_GO."
        )

    source = environment["source"]
    if source["resolved_commit"] != EXPECTED_COMMIT:
        raise RuntimeError(
            "Unexpected LongRAG source commit."
        )

    environment_config = environment[
        "environment"
    ]
    for field in (
        "install_foreign_dependencies",
        "execute_foreign_code",
        "download_models",
        "download_datasets",
        "network_access",
    ):
        if environment_config[field]:
            raise RuntimeError(
                f"Planning stage must keep {field} disabled."
            )

    if environment["retrieval_scope"][
        "generation_enabled"
    ]:
        raise RuntimeError(
            "Generation must remain disabled."
        )

    if environment["retrieval_scope"][
        "long_reader_enabled"
    ]:
        raise RuntimeError(
            "Long reader must remain disabled."
        )

    adapter_section = adapter[
        "adapter"
    ]
    if adapter_section[
        "architecture_source_commit"
    ] != EXPECTED_COMMIT:
        raise RuntimeError(
            "Adapter source commit mismatch."
        )

    if adapter_section[
        "status"
    ] != "specification_only":
        raise RuntimeError(
            "Adapter must remain specification-only."
        )

    long_unit = adapter["input_contract"]["long_unit"]
    if set(
        long_unit["required_fields"]
    ) != REQUIRED_UNIT_FIELDS:
        raise RuntimeError(
            "Long-unit field contract is incomplete."
        )

    query_contract = adapter[
        "query_contract"
    ]
    result_fields = set(
        query_contract[
            "ranked_long_unit_fields"
        ]
    )
    if result_fields != REQUIRED_RESULT_FIELDS:
        raise RuntimeError(
            "Ranked result contract is incomplete."
        )

    metrics = set(
        adapter["evaluation"][
            "common_document_metrics"
        ]
    )
    if metrics != REQUIRED_METRICS:
        raise RuntimeError(
            "Evaluation metric contract is incomplete."
        )

    if adapter["safety"][
        "final_generation_enabled"
    ]:
        raise RuntimeError(
            "Final generation must remain disabled."
        )

    if adapter["safety"][
        "foreign_code_modified"
    ]:
        raise RuntimeError(
            "Foreign code must not be modified."
        )

    if adapter["safety"][
        "frozen_baseline_modified"
    ]:
        raise RuntimeError(
            "Frozen baseline must not be modified."
        )

    print("[OK] LongRAG adapter plan")
    print(
        " Source commit:",
        source["resolved_commit"],
    )
    print(
        " Environment stage:",
        environment_config[
            "initial_execution_mode"
        ],
    )
    print(
        " Adapter status:",
        adapter_section["status"],
    )
    print(
        " Pilot dataset:",
        environment["pilot"]["dataset_id"],
    )
    print(
        " Pilot documents:",
        environment["pilot"]["document_limit"],
    )
    print(
        " Pilot queries:",
        environment["pilot"]["query_limit"],
    )
    print(
        " Generation enabled:",
        adapter["safety"][
            "final_generation_enabled"
        ],
    )


if __name__ == "__main__":
    main()
