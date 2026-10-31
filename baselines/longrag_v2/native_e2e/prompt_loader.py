"""Load and render LongRAG prompt templates."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

_PROMPT_DIR = Path(__file__).parent / "prompts"


@dataclass(frozen=True)
class PromptTemplate:
    name: str
    text: str
    sha256: str
    source_path: Path


def _looks_like_path(value: str | Path) -> bool:
    if isinstance(value, Path):
        return True
    s = str(value)
    return s.endswith(".txt") or "/" in s or "\\" in s


def load_prompt(name_or_path: str | Path) -> PromptTemplate:
    """Load a prompt template by name or by explicit path.

    Bare names resolve against ``native_e2e/prompts/<name>.txt``
    (nq, hotpotqa, qasper, multifieldqa).
    """
    if _looks_like_path(name_or_path):
        path = Path(name_or_path)
        name = path.stem
    else:
        name = str(name_or_path)
        path = _PROMPT_DIR / f"{name}.txt"

    if not path.exists():
        raise FileNotFoundError(f"Prompt template not found: {path}")

    text = path.read_text(encoding="utf-8")
    sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return PromptTemplate(name=name, text=text, sha256=sha, source_path=path)


def render(template: PromptTemplate, *, context: str, question: str) -> str:
    """Substitute {context} and {question} in the template.

    Uses ``str.replace`` rather than ``str.format`` so unrelated braces in
    the prompt text are preserved. Raises ``ValueError`` if either
    placeholder is missing from the template.
    """
    if "{context}" not in template.text:
        raise ValueError(
            f"Prompt template {template.name!r} missing '{{context}}' placeholder."
        )
    if "{question}" not in template.text:
        raise ValueError(
            f"Prompt template {template.name!r} missing '{{question}}' placeholder."
        )
    return (
        template.text
        .replace("{context}", context)
        .replace("{question}", question)
    )