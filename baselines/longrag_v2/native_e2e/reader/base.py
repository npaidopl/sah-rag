"""Reader protocol for LongRAG native end-to-end (Stage B)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Reader(Protocol):
    """A reader turns a fully-formatted prompt into a text answer.

    The prompt is assembled by the runner from a template, the retrieved
    context, and the question. This keeps readers agnostic to retrieval
    details and to the specific prompt wording.
    """

    @property
    def name(self) -> str:
        """Stable identifier used in manifests (e.g. 'openai::gpt-4o')."""
        ...

    def generate(self, prompt: str) -> str:
        """Produce an answer for the fully-formatted prompt."""
        ...