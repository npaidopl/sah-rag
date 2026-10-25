"""HippoRAG2 baseline adapter for SAH-RAG v2."""
from __future__ import annotations

import importlib.util
import inspect
import json
import logging
import pickle
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from baselines.base import (
    BaseBaseline,
    BaselineConfig,
    BaselineStatus,
    Embedder,
    LLM,
    RetrievedChunk,
    RunMode,
)

logger = logging.getLogger(__name__)

_HIPPORAG_AVAILABLE: bool = importlib.util.find_spec("hipporag") is not None


_OPENIE_PROMPT = """You are an information extraction engine. Extract named \
entities and (subject, relation, object) triples from the PASSAGE.

PASSAGE:
\"\"\"{passage}\"\"\"

Rules:
- Extract entities: people, places, organizations, concepts, events, and
  important noun phrases. Use the surface form from the passage.
- Extract triples where subject and object are entities from the passage.
- Keep the relation short (1-4 words).

Respond ONLY with valid JSON matching this schema:
{{"entities": ["entity1", "entity2"], "triples": [["subject", "relation", "object"]]}}

No prose, no code fences, no explanations. JSON only."""


_QUERY_ENTITY_PROMPT = """Extract the named entities and key concepts from the QUERY.

QUERY:
\"\"\"{query}\"\"\"

Respond ONLY with valid JSON: {{"entities": ["entity1", "entity2"]}}
No prose, no code fences, no explanations. JSON only."""


def _chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    if not text:
        return []
    if chunk_size <= 0 or len(text) <= chunk_size:
        return [text]
    step = max(1, chunk_size - overlap)
    chunks: list[str] = []
    for start in range(0, len(text), step):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
    return chunks


def _parse_json_loose(text: str) -> Any:
    if not text:
        return None
    s = text.strip()
    if s.startswith("```"):
        lines = s.split("\n")
        s = "\n".join(lines[1:])
        if s.endswith("```"):
            s = s[:-3]
        s = s.strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", s, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    return None


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / (na * nb)


def _personalized_pagerank(
    n_nodes: int,
    adj: dict[int, dict[int, float]],
    seeds: dict[int, float],
    damping: float = 0.5,
    max_iter: int = 100,
    tol: float = 1e-8,
) -> list[float]:
    if n_nodes <= 0:
        return []
    for i in range(n_nodes):
        if i not in adj or not adj[i]:
            adj[i] = {i: 1.0}

    out_weight = {u: sum(adj[u].values()) for u in range(n_nodes)}
    total_seed = sum(seeds.values())
    if total_seed <= 0.0:
        raise ValueError("personalized_pagerank: seeds must have positive weight")

    v = {i: seeds.get(i, 0.0) / total_seed for i in range(n_nodes)}
    p = dict(v)

    for _ in range(max_iter):
        new_p = {i: 0.0 for i in range(n_nodes)}
        for u in range(n_nodes):
            pu = p[u]
            if pu == 0.0:
                continue
            ow = out_weight[u]
            if ow <= 0.0:
                continue
            for w, wt in adj[u].items():
                new_p[w] += damping * pu * wt / ow
        for i in range(n_nodes):
            new_p[i] += (1.0 - damping) * v[i]
        total = sum(new_p.values())
        if total > 0.0:
            new_p = {i: val / total for i, val in new_p.items()}
        diff = sum(abs(new_p[i] - p[i]) for i in range(n_nodes))
        p = new_p
        if diff < tol:
            break
    return [p[i] for i in range(n_nodes)]


def _ensure_hf_datasets() -> bool:
    """Force HuggingFace `datasets` into sys.modules.

    The SAH-RAG project has a top-level `datasets/` package that shadows the
    HuggingFace `datasets` library. When `sentence_transformers` (used by
    hipporag's TransformersEmbeddingModel) does `from datasets import Dataset`,
    it gets the project's version which lacks the `Dataset` class.

    Fix: load HF `datasets` explicitly from site-packages and override.
    Safe for project code that has already imported `datasets.*` submodules,
    since their references are cached locally.
    """
    import importlib.util
    import site
    current = sys.modules.get("datasets")
    if current is not None and hasattr(current, "Dataset"):
        return True
    try:
        site_packages = site.getsitepackages()
    except Exception:
        return False
    for sp in site_packages:
        hf_init = Path(sp) / "datasets" / "__init__.py"
        if not hf_init.is_file():
            continue
        try:
            spec = importlib.util.spec_from_file_location(
                "datasets",
                str(hf_init),
                submodule_search_locations=[str(hf_init.parent)],
            )
            if spec is None or spec.loader is None:
                continue
            mod = importlib.util.module_from_spec(spec)
            sys.modules["datasets"] = mod
            spec.loader.exec_module(mod)
            if hasattr(mod, "Dataset"):
                return True
        except Exception as exc:
            logger.warning("Failed to force-load HF datasets: %s", exc)
            continue
    logger.error("Could not load HuggingFace datasets. Native mode may fail.")
    return False


class HippoRAG2Baseline(BaseBaseline):
    name = "hipporag2"
    version = "0.1.0"
    status = BaselineStatus.EXECUTABLE
    supports_native = _HIPPORAG_AVAILABLE
    supports_controlled = True
    is_llm_based = True

    def __init__(
        self,
        *,
        config: BaselineConfig,
        work_dir: Path,
        embedder: Embedder | None = None,
        llm: LLM | None = None,
    ) -> None:
        super().__init__(
            config=config, work_dir=work_dir, embedder=embedder, llm=llm
        )
        extra: dict[str, Any] = dict(config.extra or {})
        self._chunk_size = int(extra.get("chunk_size", 0))
        self._chunk_overlap = int(extra.get("chunk_overlap", 0))
        self._synonymy_edge_topk = int(extra.get("synonymy_edge_topk", 2047))
        self._synonymy_edge_sim_threshold = float(
            extra.get("synonymy_edge_sim_threshold", 0.8)
        )
        self._ppr_damping = float(extra.get("ppr_damping", 0.5))
        self._passage_node_weight = float(extra.get("passage_node_weight", 0.05))
        self._linking_top_k = int(extra.get("linking_top_k", 5))
        self._max_entities_per_passage = int(
            extra.get("max_entities_per_passage", 50)
        )
        self._enable_dense_fallback = bool(
            extra.get("enable_dense_fallback", True)
        )

        self._adj: dict[int, dict[int, float]] | None = None
        self._n_nodes: int = 0
        self._entity_list: list[str] = []
        self._passage_list: list[str] = []
        self._passages: list[tuple[str, str, str]] = []
        self._ent_embeddings: list[list[float]] = []
        self._pass_embeddings: list[list[float]] = []

        self._native_engine: Any = None

    def build_index(self, documents: list[Any]) -> None:
        if self.config.mode == RunMode.NATIVE:
            self._build_native(documents)
        else:
            self._build_controlled(documents)

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        k = int(top_k) if top_k is not None else self.config.top_k
        if self.config.mode == RunMode.NATIVE:
            return self._retrieve_native(query, k)
        return self._retrieve_controlled(query, k)

    def _build_native(self, documents: list[Any]) -> None:
        if not _HIPPORAG_AVAILABLE:
            raise RuntimeError(
                "Native mode requires the 'hipporag' package. "
                "Install with: pip install git+https://github.com/OSU-NLP-Group/HippoRAG.git"
            )

        _ensure_hf_datasets()

        passages: list[tuple[str, str, str]] = []
        for doc in documents:
            text = getattr(doc, "text_normalized", None) or getattr(doc, "text_raw", "")
            doc_id = getattr(doc, "doc_id", None)
            if not doc_id or not text:
                continue
            chunks = _chunk_text(text, self._chunk_size, self._chunk_overlap)
            for i, chunk in enumerate(chunks):
                chunk_id = f"{doc_id}::chunk_{i}" if len(chunks) > 1 else doc_id
                passages.append((chunk_id, doc_id, chunk))

        if not passages:
            raise ValueError("No passages to index (empty corpus).")

        seen: dict[str, tuple[str, str]] = {}
        for chunk_id, doc_id, text in passages:
            if text in seen:
                logger.warning(
                    "Duplicate passage text detected; mapping to first occurrence."
                )
                continue
            seen[text] = (chunk_id, doc_id)
        passage_texts = list(seen.keys())

        self._passages = [
            (cid, did, txt) for txt, (cid, did) in seen.items()
        ]
        meta = {
            "passages": [list(p) for p in self._passages],
            "config": {
                "ppr_damping": self._ppr_damping,
                "synonymy_edge_topk": self._synonymy_edge_topk,
                "synonymy_edge_sim_threshold": self._synonymy_edge_sim_threshold,
                "passage_node_weight": self._passage_node_weight,
                "linking_top_k": self._linking_top_k,
            },
        }
        (self.work_dir / "meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        from hipporag import HippoRAG
        from hipporag.utils.config_utils import BaseConfig

        extra: dict[str, Any] = dict(self.config.extra or {})
        config_kwargs: dict[str, Any] = {
            "save_dir": str(self.work_dir / "native_engine"),
            "llm_name": extra.get("llm_name", "gpt-4o-mini"),
            "embedding_model_name": extra.get(
                "embedder_model_name", "Transformers/BAAI/bge-m3"
            ),
            "openie_mode": extra.get("openie_mode", "online"),
            "graph_type": extra.get(
                "graph_type", "facts_and_sim_passage_node_unidirectional"
            ),
            "synonymy_edge_topk": self._synonymy_edge_topk,
            "synonymy_edge_sim_threshold": self._synonymy_edge_sim_threshold,
            "damping": self._ppr_damping,
            "linking_top_k": self._linking_top_k,
            "passage_node_weight": self._passage_node_weight,
            "force_index_from_scratch": False,
            "force_openie_from_scratch": False,
        }
        try:
            valid = set(inspect.signature(BaseConfig).parameters.keys())
            config_kwargs = {k: v for k, v in config_kwargs.items() if k in valid}
        except (ValueError, TypeError):
            pass
        base_config = BaseConfig(**config_kwargs)
        engine = HippoRAG(global_config=base_config)

        logger.info("Indexing %d passages with native HippoRAG...", len(passage_texts))
        engine.index(passage_texts)
        self._native_engine = engine

    def _retrieve_native(self, query: str, k: int) -> list[RetrievedChunk]:
        if self._native_engine is None:
            self._load_native()
        if self._native_engine is None:
            raise RuntimeError("Native engine not built; call build_index first.")
        try:
            results = self._native_engine.retrieve([query], num_to_retrieve=k)
        except Exception as exc:  # noqa: BLE001
            logger.error("Native retrieve failed: %s", exc)
            return []
        if not results or not results[0]:
            return []

        qs = results[0]
        docs = getattr(qs, "docs", None) or []
        scores = getattr(qs, "doc_scores", None)
        if scores is None or len(scores) != len(docs):
            scores = [1.0 - i / max(1, len(docs)) for i in range(len(docs))]

        text_map: dict[str, tuple[str, str]] = {}
        for chunk_id, doc_id, text in self._passages:
            text_map.setdefault(text, (chunk_id, doc_id))

        out: list[RetrievedChunk] = []
        rank = 1
        for item, score in zip(docs, scores):
            if isinstance(item, dict):
                text = item.get("text") or item.get("content") or str(item)
            else:
                text = str(item)
            if text not in text_map:
                continue
            chunk_id, doc_id = text_map[text]
            out.append(
                RetrievedChunk(
                    doc_id=doc_id,
                    chunk_id=chunk_id,
                    text=text,
                    score=float(score),
                    rank=rank,
                    parent_doc_id=doc_id if chunk_id != doc_id else None,
                )
            )
            rank += 1
            if rank > k:
                break
        return out

    def _load_native(self) -> None:
        _ensure_hf_datasets()
        meta_path = self.work_dir / "meta.json"
        if not meta_path.exists():
            return
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            self._passages = [tuple(p) for p in meta.get("passages", [])]
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to load native meta.json: %s", exc)
            return
        if not _HIPPORAG_AVAILABLE:
            return
        try:
            from hipporag import HippoRAG
            from hipporag.utils.config_utils import BaseConfig

            extra: dict[str, Any] = dict(self.config.extra or {})
            config_kwargs: dict[str, Any] = {
                "save_dir": str(self.work_dir / "native_engine"),
                "llm_name": extra.get("llm_name", "gpt-4o-mini"),
                "embedding_model_name": extra.get(
                    "embedder_model_name", "Transformers/BAAI/bge-m3"
                ),
                "openie_mode": extra.get("openie_mode", "online"),
                "force_index_from_scratch": False,
                "force_openie_from_scratch": False,
            }
            try:
                valid = set(inspect.signature(BaseConfig).parameters.keys())
                config_kwargs = {
                    k: v for k, v in config_kwargs.items() if k in valid
                }
            except (ValueError, TypeError):
                pass
            base_config = BaseConfig(**config_kwargs)
            self._native_engine = HippoRAG(global_config=base_config)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to reload native engine: %s", exc)

    def _build_controlled(self, documents: list[Any]) -> None:
        if self.embedder is None:
            raise ValueError("controlled mode requires an injected embedder")
        if self.llm is None:
            raise ValueError("controlled mode requires an injected llm")

        passages: list[tuple[str, str, str]] = []
        for doc in documents:
            text = getattr(doc, "text_normalized", None) or getattr(doc, "text_raw", "")
            doc_id = getattr(doc, "doc_id", None)
            if not doc_id or not text:
                continue
            chunks = _chunk_text(text, self._chunk_size, self._chunk_overlap)
            for i, chunk in enumerate(chunks):
                chunk_id = f"{doc_id}::chunk_{i}" if len(chunks) > 1 else doc_id
                passages.append((chunk_id, doc_id, chunk))
        if not passages:
            raise ValueError("No passages to index (empty corpus).")

        all_entities: set[str] = set()
        triples: list[tuple[str, str, str, str]] = []
        entity_to_passages: dict[str, set[str]] = defaultdict(set)
        passage_to_entities: dict[str, set[str]] = defaultdict(set)

        for chunk_id, _doc_id, text in passages:
            result = self._openie(text)
            for ent in result.get("entities", [])[: self._max_entities_per_passage]:
                norm = self._normalize_entity(ent)
                if not norm:
                    continue
                all_entities.add(norm)
                entity_to_passages[norm].add(chunk_id)
                passage_to_entities[chunk_id].add(norm)
            for t in result.get("triples", []):
                s, r, o = t
                ns = self._normalize_entity(s)
                no = self._normalize_entity(o)
                nr = str(r).strip()
                if not (ns and no and nr):
                    continue
                all_entities.add(ns)
                all_entities.add(no)
                triples.append((ns, nr, no, chunk_id))
                entity_to_passages[ns].add(chunk_id)
                entity_to_passages[no].add(chunk_id)

        entity_list = sorted(all_entities)
        entity_idx = {e: i for i, e in enumerate(entity_list)}
        passage_list = [p[0] for p in passages]
        passage_idx = {p: i for i, p in enumerate(passage_list)}
        n_entities = len(entity_list)
        n_passages = len(passage_list)
        n_nodes = n_entities + n_passages

        edges: list[tuple[int, int, float]] = []
        for s, _r, o, _cid in triples:
            si = entity_idx[s]
            oi = entity_idx[o]
            if si == oi:
                continue
            edges.append((si, oi, 1.0))
            edges.append((oi, si, 1.0))
        for entity, cids in entity_to_passages.items():
            ei = entity_idx[entity]
            for cid in cids:
                pi = n_entities + passage_idx[cid]
                edges.append((ei, pi, 1.0))

        ent_embeddings: list[list[float]] = []
        pass_embeddings: list[list[float]] = []
        if entity_list:
            ent_embeddings = self.embedder.encode_documents(entity_list)
        if passages:
            pass_embeddings = self.embedder.encode_documents(
                [p[2] for p in passages]
            )
        if ent_embeddings and n_entities > 1:
            for i in range(n_entities):
                sims: list[tuple[int, float]] = []
                for j in range(n_entities):
                    if i == j:
                        continue
                    sim = _cosine(ent_embeddings[i], ent_embeddings[j])
                    if sim >= self._synonymy_edge_sim_threshold:
                        sims.append((j, sim))
                sims.sort(key=lambda x: -x[1])
                for j, sim in sims[: self._synonymy_edge_topk]:
                    edges.append((i, j, float(sim)))

        adj: dict[int, dict[int, float]] = defaultdict(dict)
        for u, v, w in edges:
            adj[u][v] = adj[u].get(v, 0.0) + float(w)

        self._n_nodes = n_nodes
        self._entity_list = entity_list
        self._passage_list = passage_list
        self._passages = passages
        self._ent_embeddings = ent_embeddings
        self._pass_embeddings = pass_embeddings
        self._adj = dict(adj)

        meta = {
            "n_nodes": n_nodes,
            "entity_list": entity_list,
            "passage_list": passage_list,
            "passages": [list(p) for p in passages],
            "config": {
                "ppr_damping": self._ppr_damping,
                "synonymy_edge_topk": self._synonymy_edge_topk,
                "synonymy_edge_sim_threshold": self._synonymy_edge_sim_threshold,
                "passage_node_weight": self._passage_node_weight,
                "linking_top_k": self._linking_top_k,
            },
        }
        (self.work_dir / "meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        with open(self.work_dir / "graph.pkl", "wb") as f:
            pickle.dump(dict(adj), f)
        with open(self.work_dir / "entity_embeddings.pkl", "wb") as f:
            pickle.dump(ent_embeddings, f)
        with open(self.work_dir / "passage_embeddings.pkl", "wb") as f:
            pickle.dump(pass_embeddings, f)

    def _load_controlled(self) -> None:
        if self._adj is not None:
            return
        meta_path = self.work_dir / "meta.json"
        if not meta_path.exists():
            raise RuntimeError("controlled index not found; call build_index first")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self._n_nodes = int(meta["n_nodes"])
        self._entity_list = list(meta["entity_list"])
        self._passage_list = list(meta["passage_list"])
        self._passages = [tuple(p) for p in meta["passages"]]
        cfg = meta.get("config", {})
        self._ppr_damping = float(cfg.get("ppr_damping", self._ppr_damping))
        self._synonymy_edge_topk = int(
            cfg.get("synonymy_edge_topk", self._synonymy_edge_topk)
        )
        self._synonymy_edge_sim_threshold = float(
            cfg.get("synonymy_edge_sim_threshold", self._synonymy_edge_sim_threshold)
        )
        self._passage_node_weight = float(
            cfg.get("passage_node_weight", self._passage_node_weight)
        )
        self._linking_top_k = int(cfg.get("linking_top_k", self._linking_top_k))
        with open(self.work_dir / "graph.pkl", "rb") as f:
            self._adj = pickle.load(f)
        with open(self.work_dir / "entity_embeddings.pkl", "rb") as f:
            self._ent_embeddings = pickle.load(f)
        with open(self.work_dir / "passage_embeddings.pkl", "rb") as f:
            self._pass_embeddings = pickle.load(f)

    def _retrieve_controlled(self, query: str, k: int) -> list[RetrievedChunk]:
        self._load_controlled()
        if self.embedder is None:
            raise RuntimeError("controlled mode requires an injected embedder")

        query_entities = [
            self._normalize_entity(e)
            for e in self._openie_query(query).get("entities", [])
        ]
        query_entities = [e for e in query_entities if e]
        if not query_entities:
            logger.warning("No entities extracted from query; dense fallback.")
            return self._dense_fallback(query, k)

        if not self._entity_list or not self._ent_embeddings:
            logger.warning("Empty entity graph; dense fallback.")
            return self._dense_fallback(query, k)
        q_embs = self.embedder.encode_documents(query_entities)
        seeds: dict[int, float] = {}
        for q_emb in q_embs:
            best_i, best_sim = -1, -1.0
            for i, e_emb in enumerate(self._ent_embeddings):
                sim = _cosine(q_emb, e_emb)
                if sim > best_sim:
                    best_sim = sim
                    best_i = i
            if best_i >= 0 and best_sim > 0.4:
                seeds[best_i] = seeds.get(best_i, 0.0) + float(best_sim)

        if not seeds:
            logger.warning("No graph seeds matched; dense fallback.")
            return self._dense_fallback(query, k)

        n_entities = len(self._entity_list)
        n_passages = len(self._passage_list)
        total_entity_weight = sum(seeds.values())
        if n_passages > 0 and total_entity_weight > 0.0:
            per_passage = (
                self._passage_node_weight * total_entity_weight
            ) / n_passages
            for i in range(n_passages):
                seeds[n_entities + i] = per_passage

        assert self._adj is not None
        ppr = _personalized_pagerank(
            n_nodes=self._n_nodes,
            adj={k2: dict(v2) for k2, v2 in self._adj.items()},
            seeds=seeds,
            damping=self._ppr_damping,
        )

        scored: list[tuple[str, float]] = []
        for i, cid in enumerate(self._passage_list):
            scored.append((cid, float(ppr[n_entities + i])))
        scored.sort(key=lambda x: -x[1])

        chunk_to_passage = {p[0]: p for p in self._passages}
        out: list[RetrievedChunk] = []
        rank = 1
        for cid, score in scored:
            if cid not in chunk_to_passage:
                continue
            _, doc_id, text = chunk_to_passage[cid]
            out.append(
                RetrievedChunk(
                    doc_id=doc_id,
                    chunk_id=cid,
                    text=text,
                    score=score,
                    rank=rank,
                    parent_doc_id=doc_id if cid != doc_id else None,
                )
            )
            rank += 1
            if rank > k:
                break
        return out

    def _dense_fallback(self, query: str, k: int) -> list[RetrievedChunk]:
        if not self._enable_dense_fallback:
            return []
        if self.embedder is None or not self._pass_embeddings:
            return []
        q_emb = self.embedder.encode_query(query)
        scored = [
            (i, _cosine(q_emb, p_emb))
            for i, p_emb in enumerate(self._pass_embeddings)
        ]
        scored.sort(key=lambda x: -x[1])
        chunk_to_passage = {p[0]: p for p in self._passages}
        out: list[RetrievedChunk] = []
        rank = 1
        for i, sim in scored:
            if i >= len(self._passage_list):
                continue
            cid = self._passage_list[i]
            if cid not in chunk_to_passage:
                continue
            _, doc_id, text = chunk_to_passage[cid]
            out.append(
                RetrievedChunk(
                    doc_id=doc_id,
                    chunk_id=cid,
                    text=text,
                    score=float(sim),
                    rank=rank,
                    parent_doc_id=doc_id if cid != doc_id else None,
                )
            )
            rank += 1
            if rank > k:
                break
        return out

    def _openie(self, passage: str) -> dict[str, list]:
        if self.llm is None:
            return {"entities": [], "triples": []}
        try:
            raw = self.llm.generate(_OPENIE_PROMPT.format(passage=passage))
        except Exception as exc:  # noqa: BLE001
            logger.warning("LLM generate failed (OpenIE): %s", exc)
            return {"entities": [], "triples": []}
        parsed = _parse_json_loose(raw)
        if not isinstance(parsed, dict):
            return {"entities": [], "triples": []}
        entities = [
            str(e).strip() for e in parsed.get("entities", []) if str(e).strip()
        ]
        triples_raw = parsed.get("triples", [])
        triples: list[list[str]] = []
        for t in triples_raw:
            if isinstance(t, (list, tuple)) and len(t) == 3:
                s, r, o = (str(x).strip() for x in t)
                if s and r and o:
                    triples.append([s, r, o])
        return {"entities": entities, "triples": triples}

    def _openie_query(self, query: str) -> dict[str, list]:
        if self.llm is None:
            return {"entities": []}
        try:
            raw = self.llm.generate(_QUERY_ENTITY_PROMPT.format(query=query))
        except Exception as exc:  # noqa: BLE001
            logger.warning("LLM generate failed (query entities): %s", exc)
            return {"entities": []}
        parsed = _parse_json_loose(raw)
        if not isinstance(parsed, dict):
            return {"entities": []}
        entities = [
            str(e).strip() for e in parsed.get("entities", []) if str(e).strip()
        ]
        return {"entities": entities}

    @staticmethod
    def _normalize_entity(name: str) -> str:
        s = str(name).strip().lower()
        s = re.sub(r"[^\w\s\-]", "", s, flags=re.UNICODE)
        s = re.sub(r"\s+", " ", s)
        return s.strip()