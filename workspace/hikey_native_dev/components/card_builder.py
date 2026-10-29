"""Doc_card and Sec_card construction (paper Section 3.1, Appendix D.3)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .dhp_parser import Block, BLOCK_HEADER, BLOCK_TITLE


@dataclass
class SectionUnit:
    unit_id: str
    block: Block
    content: str

    @property
    def unit_type(self) -> str:
        return self.block.block_type


@dataclass
class SecCard:
    section_id: str
    doc_id: str
    title: str
    section_path: list[str]
    units: list[SectionUnit] = field(default_factory=list)


@dataclass
class DocCard:
    doc_id: str
    title: str
    section_paths: list[list[str]]
    sections: list[SecCard] = field(default_factory=list)

    @property
    def card_text(self) -> str:
        path_strs = [" > ".join(p) for p in self.section_paths if p]
        return self.title + " || " + " | ".join(path_strs)


def build_section_tree(
    blocks: list[Block], doc_id: str,
) -> DocCard:
    """Given a flat list of blocks, build a DocCard + SecCards."""
    by_id = {b.block_id: b for b in blocks}
    title_block = next(
        (b for b in blocks if b.block_type == BLOCK_TITLE), None
    )
    title_text = title_block.content if title_block else doc_id

    # Sections = every HEADER block becomes a SecCard; its children units
    # are TEXT/TABLE/FIGURE/CAPTION blocks directly under it.
    sections: list[SecCard] = []
    for b in blocks:
        if b.block_type != BLOCK_HEADER:
            continue
        sec = SecCard(
            section_id=b.block_id,
            doc_id=doc_id,
            title=b.content,
            section_path=list(b.section_path),
        )
        for child_id in b.children_ids:
            child = by_id.get(child_id)
            if child is None:
                continue
            if child.block_type == BLOCK_HEADER:
                continue
            sec.units.append(
                SectionUnit(unit_id=child.block_id, block=child, content=child.content)
            )
        sections.append(sec)

    # Any units not under a header get their own synthetic section.
    claimed: set[str] = set()
    for s in sections:
        for u in s.units:
            claimed.add(u.unit_id)
    orphan_units = [
        b for b in blocks
        if b.block_type in {"text", "table", "figure", "caption"}
        and b.block_id not in claimed
    ]
    if orphan_units:
        sec = SecCard(
            section_id="orphans",
            doc_id=doc_id,
            title="(unscoped)",
            section_path=[title_text, "(unscoped)"],
        )
        for b in orphan_units:
            sec.units.append(
                SectionUnit(unit_id=b.block_id, block=b, content=b.content)
            )
        sections.append(sec)

    section_paths = [s.section_path for s in sections if s.section_path]
    return DocCard(
        doc_id=doc_id,
        title=title_text,
        section_paths=section_paths,
        sections=sections,
    )


def upper_context(
    unit_block: Block, blocks_by_id: dict[str, Block]
) -> str:
    """Paper 3.1: 'For visual units lacking captions, HiKEY traverses the
    DHP tree to retrieve a logically preceding textual node as Upper Context.'

    We approximate: walk up to the section, take the section title, plus
    the text of the immediately preceding sibling text block (if any).
    """
    if unit_block.block_type not in {"table", "figure"}:
        return ""
    parts: list[str] = []
    if unit_block.section_path:
        parts.append(" > ".join(unit_block.section_path))
    parent = blocks_by_id.get(unit_block.parent_id or "")
    if parent:
        for cid in parent.children_ids:
            if cid == unit_block.block_id:
                break
            sib = blocks_by_id.get(cid)
            if sib and sib.block_type == "text" and sib.content:
                parts.append(sib.content[:400])
                break
    return " ".join(parts)