"""HiKEY native end-to-end pipeline.

Workflow:
    index(pdf_dir)   -> parse every PDF, build Doc/Sec cards, encode, persist
    retrieve(q, k)   -> Stage-1 routing + Stage-2 MaxSim + Stage-3 evidence
    answer(q, k)     -> retrieve + LVLM reader (optional)
"""
from __future__ import annotations

import json
import re
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .components.card_builder import DocCard, SecCard, build_section_tree, upper_context
from .components.dhp_parser import Block, parse_pdf
from .components.evidence_assembler import (
    Unit,
    assemble_evidence,
    default_token_count,
)
from .components.lvlm_reader import LVLMReader
from .components.maxsim import cosine, final_section_score, type_specific_score
from .components.multimodal_encoder import MultiModalEncoder

logger = logging.getLogger(__name__)


class HiKEYNativePipeline:
    def __init__(
        self,
        work_dir: Path | str,
        *,
        alpha: float = 0.5,
        gamma: float = 0.7,
        lam: float = 0.5,
        token_budget: int = 16000,
        doc_route_top_n: int = 30,
        mock: bool = False,
        mock_parser: bool | None = None,
        mock_encoders: bool | None = None,
        encoder: MultiModalEncoder | None = None,
        reader: LVLMReader | None = None,
    ) -> None:
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.alpha = alpha
        self.gamma = gamma
        self.lam = lam
        self.token_budget = token_budget
        self.doc_route_top_n = doc_route_top_n
        self.mock = mock
        self.mock_parser = mock if mock_parser is None else mock_parser
        self.mock_encoders = mock if mock_encoders is None else mock_encoders

        self.encoder = encoder or MultiModalEncoder(mock=self.mock_encoders)
        self.reader = reader or LVLMReader(mock=mock)

        self.docs: list[DocCard] = []
        self.doc_embeddings: list[list[float]] = []
        self.doc_tokens: list[list[str]] = []
        self.sec_units: dict[str, list[Unit]] = {}
        self.sec_unit_embeddings: dict[str, list[list[float]]] = {}
        self.sec_embeddings: dict[str, list[float]] = {}

    # ---------- indexing ----------

    def index(self, pdf_dir: Path | str) -> None:
        pdf_dir = Path(pdf_dir)
        pdfs = sorted(pdf_dir.glob("*.pdf"))
        if not pdfs and not self.mock_parser:
            raise FileNotFoundError(f"No PDFs found in {pdf_dir}")

        self.docs = []
        self.doc_embeddings = []
        self.doc_tokens = []
        self.sec_units = {}
        self.sec_unit_embeddings = {}
        self.sec_embeddings = {}

        for pdf_path in pdfs:
            doc_id = pdf_path.stem
            logger.info("Parsing %s", pdf_path)
            blocks = parse_pdf(pdf_path, mock=self.mock_parser)
            doc_card = build_section_tree(blocks, doc_id=doc_id)
            self.docs.append(doc_card)
            self._build_doc_embeddings(doc_card, blocks)

        self._persist_manifest()

    def _build_doc_embeddings(self, doc: DocCard, blocks: list[Block]) -> None:
        # Doc_card embedding
        doc_emb = self.encoder.encode_texts([doc.card_text])[0]
        self.doc_embeddings.append(doc_emb)
        self.doc_tokens.append(doc.card_text.lower().split())

        blocks_by_id = {b.block_id: b for b in blocks}

        for sec in doc.sections:
            units: list[Unit] = []
            for u in sec.units:
                # Skip noise units: empty, near-empty, or numeric-only
                content = (u.content or "").strip()
                if not content:
                    continue
                if len(content) < 3 and u.block.block_type == "text":
                    continue
                # Skip pure numbers/punctuation (table cell noise from infoboxes)
                stripped = re.sub(r"[\d\s.,:;\-–—+/()%$€£#]", "", content)
                if len(stripped) < 3:
                    continue
                unit = Unit(
                    unit_id=u.unit_id,
                    doc_id=doc.doc_id,
                    section_id=sec.section_id,
                    section_path=list(u.block.section_path),
                    unit_type=u.block.block_type,
                    content=content,
                    image_bytes=u.block.image_bytes,
                )
                units.append(unit)
            if not units:
                continue

            self.sec_units[sec.section_id] = units

            # Encode each unit
            unit_embs: list[list[float]] = []
            for unit in units:
                if unit.unit_type == "text":
                    emb = self.encoder.encode_texts([unit.content or ""])[0]
                elif unit.unit_type in ("table", "figure"):
                    if unit.image_bytes:
                        emb = self.encoder.encode_image(unit.image_bytes)
                    else:
                        emb = self.encoder.encode_texts([unit.content or ""])[0]
                else:
                    emb = self.encoder.encode_texts([unit.content or ""])[0]
                unit_embs.append(emb)
            self.sec_unit_embeddings[sec.section_id] = unit_embs

            # Section-level embedding: use the first text unit (or section title)
            sec_text = " ".join(u.content for u in units if u.unit_type == "text") or sec.title
            self.sec_embeddings[sec.section_id] = self.encoder.encode_texts([sec_text])[0]

    def _persist_manifest(self) -> None:
        manifest = {
            "n_docs": len(self.docs),
            "n_sections": len(self.sec_units),
            "n_units": sum(len(u) for u in self.sec_units.values()),
            "config": {
                "alpha": self.alpha,
                "gamma": self.gamma,
                "lam": self.lam,
                "token_budget": self.token_budget,
                "doc_route_top_n": self.doc_route_top_n,
            },
            "docs": [
                {
                    "doc_id": d.doc_id,
                    "title": d.title,
                    "n_sections": len(d.sections),
                }
                for d in self.docs
            ],
        }
        (self.work_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )

    # ---------- retrieval ----------

    def _bm25_score(self, q_tokens: list[str], doc_tokens: list[str]) -> float:
        if not q_tokens or not doc_tokens:
            return 0.0
        from collections import Counter
        import math
        k1, b = 1.5, 0.75
        tf = Counter(doc_tokens)
        dl = len(doc_tokens)
        avgdl = max(sum(len(t) for t in self.doc_tokens) / max(len(self.doc_tokens), 1), 1e-9)
        score = 0.0
        for t in set(q_tokens):
            df = sum(1 for dt in self.doc_tokens if t in dt)
            if df == 0:
                continue
            idf = math.log(1.0 + (len(self.doc_tokens) - df + 0.5) / (df + 0.5))
            f = tf.get(t, 0)
            denom = f + k1 * (1 - b + b * dl / avgdl)
            if denom > 0:
                score += idf * f * (k1 + 1) / denom
        return score

    def _mm(self, scores: list[float]) -> list[float]:
        if not scores:
            return []
        lo, hi = min(scores), max(scores)
        if hi == lo:
            return [0.0] * len(scores)
        return [(s - lo) / (hi - lo) for s in scores]

    def retrieve(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        if not self.docs:
            return []
        q_tokens = query.lower().split()
        q_emb = self.encoder.encode_texts([query])[0]

        # Stage 1: document routing
        lex_scores = [self._bm25_score(q_tokens, dt) for dt in self.doc_tokens]
        den_scores = [cosine(q_emb, de) for de in self.doc_embeddings]
        lex_mm = self._mm(lex_scores)
        den_mm = self._mm(den_scores)
        doc_scores = [
            self.alpha * lex_mm[i] + (1 - self.alpha) * den_mm[i]
            for i in range(len(self.docs))
        ]
        ranked_docs = sorted(range(len(self.docs)), key=lambda i: doc_scores[i], reverse=True)
        routed = ranked_docs[: self.doc_route_top_n]

        # Stage 2: section MaxSim
        sec_candidates: list[tuple[float, str, str]] = []  # (score, doc_id, sec_id)
        for di in routed:
            doc = self.docs[di]
            for sec in doc.sections:
                unit_embs = self.sec_unit_embeddings.get(sec.section_id, [])
                if not unit_embs:
                    continue
                # Type-specific + MaxSim
                unit_scores = []
                for u, ue in zip(self.sec_units[sec.section_id], unit_embs):
                    sc = cosine(q_emb, ue)
                    u.score = sc  # persist score for downstream assembly
                    unit_scores.append(sc)
                sec_score = max(unit_scores) if unit_scores else 0.0
                final = final_section_score(doc_scores[di], sec_score, lam=self.lam)
                sec_candidates.append((final, doc.doc_id, sec.section_id))

        sec_candidates.sort(key=lambda x: x[0], reverse=True)
        ranked_sections = [
            (sid, self.sec_units[sid])
            for _, _, sid in sec_candidates[:top_k]
        ]

        # Stage 3: evidence assembly
        subgraph = assemble_evidence(
            ranked_sections,
            all_units_by_section=self.sec_units,
            similarity_lookup=lambda a, b: cosine(
                self.encoder.encode_texts([a.content or ""])[0],
                self.encoder.encode_texts([b.content or ""])[0],
            ) if (a.content and b.content) else 0.0,
            token_budget=self.token_budget,
            token_count=default_token_count,
        )

        # Return as RetrievedChunk-compatible dicts
        out: list[dict[str, Any]] = []
        for rank, unit in enumerate(subgraph.units, start=1):
            out.append({
                "doc_id": unit.doc_id,
                "chunk_id": unit.unit_id,
                "text": unit.content or "",
                "score": float(unit.score),
                "rank": rank,
                "parent_doc_id": None,
                "metadata": {
                    "section_path": unit.section_path,
                    "unit_type": unit.unit_type,
                    "source": "hikey_native_paper_exact",
                },
            })
        return out

    def answer(self, query: str, top_k: int = 10) -> dict[str, Any]:
        chunks = self.retrieve(query, top_k=top_k)
        context = "\n\n".join(
            f"[{c['doc_id']}:{c['chunk_id']}] {c['text']}" for c in chunks
        )
        try:
            response = self.reader.generate(query, context)
        except RuntimeError as e:
            response = f"[LVLM_UNAVAILABLE] {e}"
        return {"query": query, "chunks": chunks, "answer": response}