"""Safe structural inspection of canonical records."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from typing import Any


def value_type_name(value: Any) -> str:
    """Return a stable descriptive type name."""

    if value is None:
        return "null"

    if isinstance(value, bool):
        return "boolean"

    if isinstance(value, int):
        return "integer"

    if isinstance(value, float):
        return "number"

    if isinstance(value, str):
        return "string"

    if isinstance(value, dict):
        return "object"

    if isinstance(value, list):
        return "array"

    return type(value).__name__


def collect_key_types(
    records: Iterable[dict[str, Any]],
    sample_limit: int,
) -> dict[str, dict[str, int]]:
    """Collect top-level key and type frequencies."""

    counters: dict[str, Counter[str]] = {}

    for index, record in enumerate(records):
        if index >= sample_limit:
            break

        for key, value in record.items():
            counters.setdefault(key, Counter())
            counters[key][value_type_name(value)] += 1

    return {
        key: dict(sorted(counter.items()))
        for key, counter in sorted(counters.items())
    }


def describe_nested_value(
    value: Any,
    *,
    max_depth: int = 3,
    current_depth: int = 0,
) -> Any:
    """Describe nested structure without exposing full text values."""

    type_name = value_type_name(value)

    if current_depth >= max_depth:
        return {"type": type_name}

    if isinstance(value, dict):
        return {
            "type": "object",
            "keys": {
                key: describe_nested_value(
                    nested_value,
                    max_depth=max_depth,
                    current_depth=current_depth + 1,
                )
                for key, nested_value in sorted(value.items())
            },
        }

    if isinstance(value, list):
        element_types = sorted(
            {value_type_name(element) for element in value}
        )

        description: dict[str, Any] = {
            "type": "array",
            "length": len(value),
            "element_types": element_types,
        }

        if value:
            description["first_element_structure"] = describe_nested_value(
                value[0],
                max_depth=max_depth,
                current_depth=current_depth + 1,
            )

        return description

    if isinstance(value, str):
        return {
            "type": "string",
            "character_length": len(value),
        }

    return {"type": type_name}


def describe_record_structure(
    record: dict[str, Any] | None,
) -> dict[str, Any]:
    """Describe one record without copying sensitive or long content."""

    if record is None:
        return {"record": None}

    return {
        key: describe_nested_value(value)
        for key, value in sorted(record.items())
    }