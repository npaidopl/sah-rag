"""Document Hierarchical Parsing (DHP).

Per the paper (Section 3.1): "we use the Document Hierarchical Parsing
(DHP) module from M3DocDep (Shin et al., 2026) ... to reconstruct a
logical hierarchy tree T(d) from page-level layout blocks. Each block
(text, table, figure) becomes a node, connected by parent-child tree
edges, and receives an ancestor section path (e.g., Title > Section 5 > 5.3)."

Backend priority (first available wins):
    1. M3DocDep (exact upstream module from the paper)
    2. DocLayout-YOLO (layout detection, then heuristic hierarchy)
    3. PyMuPDF heuristic (font-size based, always available)
    4. Mock (synthetic tree, for tests only)
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _normalize_text(text: str) -> str:
    """Normalize PyMuPDF text: nbsp->space, collapse whitespace."""
    if not text:
        return ""
    text = text.replace("\xa0", " ").replace("\u00a0", " ")
    text = " ".join(text.split())
    return text.strip()


def _is_noise_header(text: str) -> bool:
    """Skip bullet-only or ultra-short pseudo-headers.

    Relaxed: only filter pure bullets/separators, NOT short legit headers
    like "Exhibit 10.33" or "Section 4.1". Keep anything >= 3 chars unless
    it is entirely punctuation/bullets.
    """
    t = _normalize_text(text)
    if not t:
        return True
    # Pure bullet or separator only
    if all(c in ".,;:•-—–*·|/()[]{}<>  " for c in t):
        return True
    # Single bullet char
    if len(t) <= 2 and t in {"•", "-", "*"}:
        return True
    return False


def _is_noise_text(text: str) -> bool:
    """Skip blocks with < 3 non-whitespace chars after normalization."""
    t = _normalize_text(text)
    return len(t) < 3


BLOCK_TEXT = "text"
BLOCK_TABLE = "table"
BLOCK_FIGURE = "figure"
BLOCK_HEADER = "header"
BLOCK_TITLE = "title"
BLOCK_CAPTION = "caption"


@dataclass
class Block:
    block_id: str
    block_type: str
    content: str
    page_num: int = 0
    font_size: float = 0.0
    is_bold: bool = False
    bbox: tuple[float, float, float, float] | None = None
    image_bytes: bytes | None = None
    parent_id: str | None = None
    children_ids: list[str] = field(default_factory=list)
    section_path: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "block_id": self.block_id,
            "block_type": self.block_type,
            "content": self.content,
            "page_num": self.page_num,
            "font_size": self.font_size,
            "is_bold": self.is_bold,
            "bbox": list(self.bbox) if self.bbox else None,
            "parent_id": self.parent_id,
            "children_ids": self.children_ids,
            "section_path": self.section_path,
        }


def _hash_id(s: str) -> str:
    return "b_" + hashlib.sha256(s.encode("utf-8")).hexdigest()[:12]


def _mock_parse(pdf_path: Path) -> list[Block]:
    """Return a deterministic synthetic tree for tests."""
    doc_title = pdf_path.stem.replace("_", " ").title()
    blocks: list[Block] = []
    title = Block(
        block_id=_hash_id(str(pdf_path) + ":title"),
        block_type=BLOCK_TITLE,
        content=doc_title,
        page_num=0,
        section_path=[doc_title],
    )
    blocks.append(title)

    sections = [
        ("Introduction", "This document introduces the core topic."),
        ("Method", "The method section describes the approach."),
        ("Results", "Results show strong performance."),
    ]
    for i, (sec_title, sec_text) in enumerate(sections):
        header = Block(
            block_id=_hash_id(str(pdf_path) + f":h{i}"),
            block_type=BLOCK_HEADER,
            content=sec_title,
            page_num=0,
            parent_id=title.block_id,
            section_path=[doc_title, sec_title],
        )
        blocks.append(header)
        title.children_ids.append(header.block_id)

        text = Block(
            block_id=_hash_id(str(pdf_path) + f":t{i}"),
            block_type=BLOCK_TEXT,
            content=sec_text,
            page_num=0,
            parent_id=header.block_id,
            section_path=[doc_title, sec_title],
        )
        blocks.append(text)
        header.children_ids.append(text.block_id)

    return blocks


def _pymupdf_parse(pdf_path: Path) -> list[Block]:
    """Heuristic parse via PyMuPDF (fonts + bbox)."""
    try:
        import pymupdf as fitz  # type: ignore  # PyMuPDF>=2.0
    except ImportError:
        try:
            import fitz  # type: ignore  # legacy <2.0
        except ImportError as e:
            raise RuntimeError(
                "PyMuPDF not installed. Install with: pip install PyMuPDF"
            ) from e

    doc = fitz.open(str(pdf_path))
    raw: list[dict[str, Any]] = []
    for page_num, page in enumerate(doc):
        page_dict = page.get_text("dict")
        for block in page_dict.get("blocks", []):
            if block.get("type") == 0:
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        text = span.get("text", "").strip()
                        if not text:
                            continue
                        normalized = _normalize_text(text)
                        if len(normalized) < 2:
                            continue
                        raw.append({
                            "type": "text",
                            "content": normalized,
                            "page": page_num,
                            "size": float(span.get("size", 10.0)),
                            "bold": bool(span.get("flags", 0) & 16),
                            "bbox": tuple(span.get("bbox", (0, 0, 0, 0))),
                        })
            elif block.get("type") == 1:
                raw.append({
                    "type": "figure",
                    "content": "",
                    "page": page_num,
                    "size": 0.0,
                    "bold": False,
                    "bbox": tuple(block.get("bbox", (0, 0, 0, 0))),
                    "image": block.get("image"),
                })
    doc.close()

    sizes = [r["size"] for r in raw if r["size"] > 0]
    median = sorted(sizes)[len(sizes) // 2] if sizes else 10.0

    blocks: list[Block] = []
    title_text = pdf_path.stem.replace("_", " ").title()
    title = Block(
        block_id=_hash_id(str(pdf_path) + ":title"),
        block_type=BLOCK_TITLE,
        content=title_text,
        section_path=[title_text],
    )
    blocks.append(title)

    current_header: Block = title
    path_stack: list[tuple[float, Block]] = [(float("inf"), title)]

    for r in raw:
        is_header = (
            r["type"] == "text"
            and (r["bold"] or r["size"] > median * 1.15)
            and len(r["content"]) < 200
            and not _is_noise_header(r["content"])
        )
        if is_header:
            while path_stack and path_stack[-1][0] <= r["size"]:
                path_stack.pop()
            parent = path_stack[-1][1] if path_stack else title
            new_path = list(parent.section_path) + [r["content"]]
            hdr = Block(
                block_id=_hash_id(str(pdf_path) + f":h:{r['page']}:{r['content'][:40]}"),
                block_type=BLOCK_HEADER,
                content=r["content"],
                page_num=r["page"],
                font_size=r["size"],
                is_bold=r["bold"],
                bbox=r["bbox"],
                parent_id=parent.block_id,
                section_path=new_path,
            )
            blocks.append(hdr)
            parent.children_ids.append(hdr.block_id)
            path_stack.append((r["size"], hdr))
            current_header = hdr
            continue

        btype = (
            BLOCK_FIGURE if r["type"] == "figure"
            else BLOCK_TEXT
        )
        blk = Block(
            block_id=_hash_id(str(pdf_path) + f":{r['page']}:{r['type']}:{r['content'][:40]}:{r['bbox']}"),
            block_type=btype,
            content=r["content"],
            page_num=r["page"],
            font_size=r["size"],
            is_bold=r["bold"],
            bbox=r["bbox"],
            image_bytes=r.get("image"),
            parent_id=current_header.block_id,
            section_path=list(current_header.section_path),
        )
        blocks.append(blk)
        current_header.children_ids.append(blk.block_id)

    return blocks


def _try_m3docdep(pdf_path: Path) -> list[Block] | None:
    try:
        import m3docdep  # type: ignore  # noqa: F401
    except ImportError:
        return None
    logger.warning("M3DocDep importable but adapter not implemented; falling through.")
    return None


def _try_doclayout(pdf_path: Path) -> list[Block] | None:
    try:
        import doclayout_yolo  # type: ignore  # noqa: F401
    except ImportError:
        return None
    logger.warning("DocLayout-YOLO importable but adapter not implemented; falling through.")
    return None


def parse_pdf(pdf_path: Path | str, mock: bool = False) -> list[Block]:
    """Parse a PDF into a flat list of Blocks with parent/child links.

    Returns blocks in reading order. Reconstruct the tree by following
    ``parent_id`` / ``children_ids``. Use ``build_section_tree`` from
    ``card_builder`` for higher-level grouping.
    """
    pdf_path = Path(pdf_path)
    if mock:
        return _mock_parse(pdf_path)
    for backend in (_try_m3docdep, _try_doclayout):
        result = backend(pdf_path)
        if result is not None:
            return result
    return _pymupdf_parse(pdf_path)