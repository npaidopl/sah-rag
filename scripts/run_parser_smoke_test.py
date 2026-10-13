"""Run the deterministic parser on the synthetic fixture."""

from __future__ import annotations

import json
from pathlib import Path

from rag_system.canonical.models import CanonicalDocument
from rag_system.parsing import DeterministicStructuralParser

FIXTURE_PATH = Path(
    "tests/fixtures/structured_document.json"
)

OUTPUT_PATH = Path(
    "outputs/parser_smoke_test.json"
)


def main() -> None:
    """Parse the fixture and write one artifact."""

    document = CanonicalDocument.model_validate_json(
        FIXTURE_PATH.read_text(encoding="utf-8-sig")
    )

    parser = DeterministicStructuralParser()
    parsed = parser.parse(document)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PATH.write_text(
        json.dumps(
            parsed.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print("[OK] Document:", parsed.doc_id)
    print("[OK] Parents:", len(parsed.parents))

    for parent in parsed.parents:
        print(
            "[PARENT]",
            parent.sibling_index,
            parent.parent_id,
            repr(parent.title),
            parent.normalized_char_span.start,
            parent.normalized_char_span.end,
        )

    print("[OK] Written:", OUTPUT_PATH.resolve())


if __name__ == "__main__":
    main()