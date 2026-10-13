import json
from pathlib import Path

report_path = Path("outputs/canonical_schema_report.json")
report = json.loads(report_path.read_text(encoding="utf-8"))

print("CANONICAL SCHEMA SUMMARY")
print("Root:", report["canonical_root"])
print("Dataset count:", report["dataset_count"])
print("Dataset IDs:", ", ".join(report["dataset_ids"]))
print()

for dataset_id, dataset in report["datasets"].items():
    print("=" * 72)
    print("DATASET:", dataset_id)
    print("Complete:", dataset["complete"])

    print(
        "Document keys:",
        ", ".join(dataset.get("document_top_level_types", {}).keys()),
    )

    print(
        "Example keys:",
        ", ".join(dataset.get("example_top_level_types", {}).keys()),
    )

    first_document = dataset.get("first_document_structure", {})
    first_example = dataset.get("first_example_structure", {})

    print(
        "First document structural keys:",
        ", ".join(first_document.keys()),
    )

    print(
        "First example structural keys:",
        ", ".join(first_example.keys()),
    )

    query_structure = first_example.get("query")
    ground_truth_structure = first_example.get("ground_truth")

    if query_structure:
        query_keys = (
            query_structure
            .get("keys", {})
            .keys()
        )
        print(
            "Nested query keys:",
            ", ".join(query_keys),
        )

    if ground_truth_structure:
        ground_truth_keys = (
            ground_truth_structure
            .get("keys", {})
            .keys()
        )
        print(
            "Nested ground-truth keys:",
            ", ".join(ground_truth_keys),
        )

    print()