from pathlib import Path

from rag_system.artifacts import ParsedDocument
from rag_system.canonical.jsonl import iter_jsonl

path = Path("outputs/clerk_parsed_sample.jsonl")

for raw_record in iter_jsonl(path):
    document = ParsedDocument.model_validate(raw_record)

    print("=" * 80)
    print("DOCUMENT:", document.doc_id)
    print("PARENTS:", len(document.parents))
    for parent in document.parents[:5]:
        preview = " ".join(
            parent.text_normalized.split()
        )[:160]

        print(
            f"  [{parent.sibling_index}] "
            f"title={parent.title!r} "
            f"span=({parent.normalized_char_span.start}, "
            f"{parent.normalized_char_span.end}) "
            f"preview={preview!r}"
        )

    if len(document.parents) > 5:
        print(
            "  ...",
            len(document.parents) - 5,
            "additional parents",
        )