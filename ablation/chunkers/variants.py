"""Synthetic ablation parsers: A1, A7, A9.

Each subclass of DeterministicStructuralParser uses a distinct parser_id,
yielding a distinct parser_config_hash and therefore distinct parent_ids
and downstream chunk_ids. That is required so variants can coexist.
"""
import hashlib
import random

from sah_rag.parsing.deterministic import (
    DeterministicStructuralParser,
    DeterministicParserConfig,
    ParsedDocument,
)
from sah_rag.parsing.segmentation import iter_paragraph_spans
from sah_rag.parsing.blocks import TextBlock
from sah_rag.canonical.models import CanonicalDocument

try:
    from sah_rag.artifacts.identity import build_parent_id, sha256_fingerprint
except ImportError:
    from sah_rag.artifacts.fingerprints import build_parent_id, sha256_fingerprint


class _AblationParserBase(DeterministicStructuralParser):
    PARSER_ID = "ablation_parser"
    PARSER_VERSION = "0.1.0"

    def __init__(self, **cfg_overrides):
        cfg = DeterministicParserConfig(
            parser_id=self.PARSER_ID,
            parser_version=self.PARSER_VERSION,
            **cfg_overrides,
        )
        super().__init__(config=cfg)

    def _make_parsed_document(self, document, parents):
        return ParsedDocument(
            doc_id=document.doc_id,
            dataset=document.dataset,
            parser_id=self.config.parser_id,
            parser_version=self.config.parser_version,
            parser_config_hash=self.config_hash,
            canonical_text_normalized_length=len(document.text_normalized),
            canonical_text_raw_length=len(document.text_raw),
            parents=parents,
            source_fingerprint=sha256_fingerprint({
                "doc_id": document.doc_id,
                "dataset": document.dataset,
                "text_raw": document.text_raw,
                "text_normalized": document.text_normalized,
            }),
            metadata={
                "parent_count": len(parents),
                "parser_mode": self.config.parser_id,
            },
        )


class FlatHybridParser(_AblationParserBase):
    """A1: one parent = whole document (flat)."""
    PARSER_ID = "ablation_flat_hybrid"

    def parse(self, document: CanonicalDocument) -> ParsedDocument:
        blocks = iter_paragraph_spans(document.text_normalized)
        if not blocks:
            blocks = (TextBlock(block_index=0, text="", start=0, end=0, is_heading=False),)
        parents = (self._build_parent(
            document=document, parent_index=0, blocks=blocks,
            is_first=True, is_last=True,
        ),)
        return self._make_parsed_document(document, parents)


class ParagraphParentsParser(_AblationParserBase):
    """A7: one parent per paragraph."""
    PARSER_ID = "ablation_paragraph_parents"

    def parse(self, document: CanonicalDocument) -> ParsedDocument:
        blocks = iter_paragraph_spans(document.text_normalized)
        if not blocks:
            blocks = (TextBlock(block_index=0, text="", start=0, end=0, is_heading=False),)
        parents = tuple(
            self._build_parent(
                document=document, parent_index=i, blocks=(blk,),
                is_first=(i == 0), is_last=(i == len(blocks) - 1),
            )
            for i, blk in enumerate(blocks)
        )
        return self._make_parsed_document(document, parents)


class PlaceboBoundariesParser(_AblationParserBase):
    """A9: random boundaries with size distribution from real parser."""
    PARSER_ID = "ablation_placebo_boundaries"

    def __init__(self, seed: int = 42, **cfg_overrides):
        super().__init__(**cfg_overrides)
        self._seed = seed
        self._real_parser = DeterministicStructuralParser()

    def _rng_for(self, doc_id: str) -> random.Random:
        h = hashlib.sha256(f"{self._seed}:{doc_id}".encode("utf-8")).hexdigest()
        return random.Random(int(h[:16], 16))

    def parse(self, document: CanonicalDocument) -> ParsedDocument:
        text = document.text_normalized
        if not text:
            return self._make_parsed_document(document, ())

        real = self._real_parser.parse(document)
        real_sizes = [max(20, len(p.text_normalized)) for p in real.parents]
        if not real_sizes:
            return self._make_parsed_document(document, ())

        rng = self._rng_for(document.doc_id)
        sizes = list(real_sizes)
        rng.shuffle(sizes)
        total = sum(sizes)
        text_len = len(text)
        scaled = [max(20, int(s * text_len / total)) for s in sizes]
        residual = text_len - sum(scaled)
        scaled[-1] += residual
        if scaled[-1] < 20:
            need = 20 - scaled[-1]
            scaled[-1] = 20
            scaled[0] -= need

        boundaries = [0]
        for s in scaled[:-1]:
            boundaries.append(min(boundaries[-1] + s, text_len))
        boundaries.append(text_len)

        blocks = [
            TextBlock(block_index=i, text=text[s:e], start=s, end=e, is_heading=False)
            for i, (s, e) in enumerate(zip(boundaries[:-1], boundaries[1:]))
            if e > s
        ]
        if not blocks:
            return self._make_parsed_document(document, ())

        parents = tuple(
            self._build_parent(
                document=document, parent_index=i, blocks=(blk,),
                is_first=(i == 0), is_last=(i == len(blocks) - 1),
            )
            for i, blk in enumerate(blocks)
        )
        return self._make_parsed_document(document, parents)