"""Load and render LongRAG prompt templates.

All native prompts are byte-faithful transcriptions from
utils/gpt_inference.py at pinned commit 778454e2986c39f2630fd8832abb44a5c9bc124f
(TIGER-AI-Lab/LongRAG, MIT license).

Templates are plain text files with placeholders:

    nq.txt                : {titles}, {context}, {question}
    hotpotqa.txt          : {titles}, {context}, {question}
    answer_extraction.txt : {demos},  {question}, {long_answer}

Demo examples are loaded from HuggingFace (TIGER-Lab/LongRAG subset
answer_extract_example) at runtime, cached at prompts/answer_extraction_demos.txt.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_PROMPT_DIR = Path(__file__).parent / "prompts"
_PLACEHOLDER_RE = re.compile(r"\{(\w+)\}")


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

    Bare names resolve against ``native_e2e/prompts/<name>.txt``.
    Only native-faithful names are supported: 'nq', 'hotpotqa',
    'answer_extraction'. Anything else raises FileNotFoundError.
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


def _format_titles(titles: list[str]) -> str:
    """Format titles as a Python list of double-quoted strings.

    Matches official code: titles = ['"' + title + '"' for title in titles]
    then f-string interpolation of the resulting list.
    """
    return str(['"' + str(t) + '"' for t in titles])


def render(template: PromptTemplate, **kwargs: Any) -> str:
    """Replace all placeholders in the template with provided values.

    Supported placeholders (all optional, but must be provided if the
    template contains them):

        {context}      : str
        {question}     : str
        {titles}       : list[str]  (formatted as ``['"t1"', '"t2"']``)
        {long_answer}  : str
        {demos}        : str

    Raises ValueError if any placeholder remains unresolved after
    substitution. This catches accidentally mismatched templates.
    """
    text = template.text
    for key, value in kwargs.items():
        placeholder = "{" + key + "}"
        if placeholder not in text:
            continue
        if key == "titles" and isinstance(value, list):
            formatted = _format_titles(value)
        else:
            formatted = str(value)
        text = text.replace(placeholder, formatted)

    remaining = _PLACEHOLDER_RE.findall(text)
    if remaining:
        raise ValueError(
            f"Template {template.name!r} has unresolved placeholders: "
            f"{sorted(set(remaining))}. "
            f"Provided kwargs: {sorted(kwargs.keys())}"
        )
    return text


# ---------------------------------------------------------------------------
# Demo examples (loaded from HF, cached to disk)
# ---------------------------------------------------------------------------

_DEMOS_PATH = _PROMPT_DIR / "answer_extraction_demos.txt"
_HF_DATASET = "TIGER-Lab/LongRAG"
_HF_SUBSET = "answer_extract_example"
_DEFAULT_NUM_DEMOS = 8


def _fetch_demos_from_hf(num_demo: int = _DEFAULT_NUM_DEMOS) -> str:
    """Load demo examples from HuggingFace.

    Returns the full demo prompt block starting with 'Here are some examples: '
    and ending with '\\n\\n'. Raises if HF is unavailable.
    """
    from datasets import load_dataset

    ds = load_dataset(_HF_DATASET, _HF_SUBSET)["train"]
    demo_prompt = "Here are some examples: "
    for item in ds.select(range(num_demo)):
        for answer in item["answers"]:
            demo_prompt += (
                f"Question: {item['question']}\n"
                f"Long Answer: {item['long_answer']}\n"
                f"Short Answer: {answer}\n\n"
            )
    return demo_prompt


def regenerate_demos(num_demo: int = _DEFAULT_NUM_DEMOS) -> Path:
    """Fetch demos from HuggingFace and cache them to disk. Returns path."""
    demo_prompt = _fetch_demos_from_hf(num_demo)
    _DEMOS_PATH.write_text(demo_prompt, encoding="utf-8", newline="")
    logger.info("Wrote %d chars of demos to %s", len(demo_prompt), _DEMOS_PATH)
    return _DEMOS_PATH


def load_demo_examples(
    num_demo: int = _DEFAULT_NUM_DEMOS,
    *,
    use_cache: bool = True,
) -> str:
    """Load demo examples for the answer extraction prompt.

    Resolution order:
        1. If use_cache=True and prompts/answer_extraction_demos.txt
           exists, return its contents.
        2. Otherwise, fetch from HuggingFace and cache to disk.
        3. If HF fails, raise RuntimeError.
    """
    if use_cache and _DEMOS_PATH.exists():
        return _DEMOS_PATH.read_text(encoding="utf-8")

    try:
        return _fetch_demos_from_hf(num_demo)
    except Exception as exc:
        raise RuntimeError(
            f"Could not load demo examples from HF ({_HF_DATASET}/{_HF_SUBSET}): "
            f"{exc}. Local cache at {_DEMOS_PATH} is also missing. "
            "Either restore network access or run "
            "regenerate_demos() from a machine with HF access."
        ) from exc