"""Ancestry-aware evidence subgraph assembly (paper Section 3.3,
Appendix E, Algorithm 1).

Three phases per anchor (highest-ranked section):
    1. Anchor & Ancestry  - insert anchor unit + governing headers
    2. Sibling Packing    - add sibling units under the same parent
    3. Semantic Packing   - add top-M similar units from elsewhere

Each phase is gated by the token budget. Token counting is
approximate (whitespace-split) and pluggable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable


def default_token_count(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text.split()))


@dataclass
class Unit:
    unit_id: str
    doc_id: str
    section_id: str
    section_path: list[str]
    unit_type: str          # "text" | "table" | "figure" | "caption"
    content: str
    score: float = 0.0
    image_bytes: bytes | None = None

    def serialize(self) -> str:
        path = " > ".join(self.section_path) if self.section_path else "(root)"
        header = f"[UNIT id={self.unit_id} type={self.unit_type}] path: {path}"
        body = self.content or ""
        return header + "\n" + body


@dataclass
class EvidenceSubgraph:
    units: list[Unit] = field(default_factory=list)
    total_tokens: int = 0
    token_budget: int = 0
    phases: dict[str, int] = field(default_factory=dict)

    def serialize(self) -> str:
        return "\n\n---\n\n".join(u.serialize() for u in self.units)


def assemble_evidence(
    ranked_sections: list[tuple[str, list[Unit]]],
    *,
    all_units_by_section: dict[str, list[Unit]],
    similarity_lookup: Callable[[Unit, Unit], float] | None = None,
    token_budget: int = 16000,
    max_anchors: int = 10,
    semantic_candidates: int = 20,
    token_count: Callable[[str], int] = default_token_count,
) -> EvidenceSubgraph:
    """Assemble evidence following Algorithm 1.

    Parameters
    ----------
    ranked_sections : list of (section_id, [units sorted by score desc])
    all_units_by_section : lookup for sibling expansion
    similarity_lookup : function (anchor, candidate) -> float
        Used in Phase 3 for semantic packing. If None, phase 3 is skipped.
    """
    subgraph = EvidenceSubgraph(token_budget=token_budget)
    seen: set[str] = set()

    def try_add(u: Unit) -> bool:
        cost = token_count(u.serialize())
        if subgraph.total_tokens + cost > token_budget:
            return False
        if u.unit_id in seen:
            return False
        subgraph.units.append(u)
        subgraph.total_tokens += cost
        seen.add(u.unit_id)
        return True

    phase1 = phase2 = phase3 = 0

    for i, (section_id, units) in enumerate(ranked_sections[:max_anchors]):
        if not units:
            continue
        anchor = units[0]

        # Phase 1: anchor + governing headers (implicit in serialize)
        if not try_add(anchor):
            continue
        phase1 += 1

        # Phase 2: siblings under same parent section
        siblings = all_units_by_section.get(section_id, [])
        for s in siblings:
            if s.unit_id == anchor.unit_id:
                continue
            if s.unit_type not in ("table", "figure"):
                continue
            if try_add(s):
                phase2 += 1

        # Phase 3: semantic associates
        if similarity_lookup is not None:
            candidates: list[tuple[float, Unit]] = []
            for other_sid, other_units in all_units_by_section.items():
                if other_sid == section_id:
                    continue
                for ou in other_units:
                    if ou.unit_type not in ("table", "figure"):
                        continue
                    sim = similarity_lookup(anchor, ou)
                    candidates.append((sim, ou))
            candidates.sort(key=lambda x: x[0], reverse=True)
            for _, cand in candidates[:semantic_candidates]:
                if try_add(cand):
                    phase3 += 1
                    if subgraph.total_tokens >= token_budget:
                        break

        if subgraph.total_tokens >= token_budget:
            break

    subgraph.phases = {"phase1": phase1, "phase2": phase2, "phase3": phase3}
    return subgraph