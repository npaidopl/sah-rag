"""RAPTOR baseline adapter for SAH-RAG v2.

Fidelity:
  AUTHOR_SUPPORTED_NATIVE: recursive tree + pluggable components
    (BGE-M3 + local HF LM via official interfaces). No API keys.
  PAPER_EXACT_NATIVE: SBERT + gpt-3.5-turbo + gpt-4 QA (QA unused in v2).
    Requires OPENAI_API_KEY; raises immediately without it (no silent fallback).

Loading policy: local_files_only=True by default (allow_download=False).
Prevents accidental network fetches. Pass allow_download=true to opt in.

Retrieval: collapsed tree (paper-preferred; Sec 4.2 reports +2-4.5pp over traversal).
"""
from __future__ import annotations
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from baselines.base import (
    BaseBaseline, BaselineConfig, BaselineStatus,
    Embedder, LLM, RetrievedChunk, RunMode,
)

log = logging.getLogger(__name__)

_DEFAULT_CHUNK_SIZE = 100
_DEFAULT_CHUNK_OVERLAP = 20
_DEFAULT_MAX_LEVELS = 4
_DEFAULT_CLUSTER_SIZE = 8
_DEFAULT_UMAP_COMPONENTS = 32
_DEFAULT_RANDOM_SEED = 42
_UMAP_MIN_NODES = 30

_AUTHOR_SUPPORTED = "AUTHOR_SUPPORTED_NATIVE"
_PAPER_EXACT = "PAPER_EXACT_NATIVE"
_PAPER_EXACT_SBERT_MODEL = "sentence-transformers/multi-qa-mpnet-base-cos-v1"
_PAPER_EXACT_SUMMARIZER = "gpt-3.5-turbo"


class BaseSummarizationModel(Protocol):
    def summarize(self, context: str, max_tokens: int = 200) -> str: ...


class BaseEmbeddingModel(Protocol):
    def encode(self, texts: list[str]) -> list[list[float]]: ...
    def encode_query(self, text: str) -> list[float]: ...


class _EmbedderWrapper:
    def __init__(self, embedder: Embedder) -> None:
        self._embedder = embedder
    def encode(self, texts): return self._embedder.encode_documents(texts)
    def encode_query(self, text): return self._embedder.encode_query(text)


class _LLMSummarizer:
    _PROMPT = (
        "You are a careful summarizer. Summarize the following passages "
        "into a concise, faithful abstract of at most {max_tokens} tokens. "
        "Preserve all entities, numbers, and factual claims. Do not add "
        "information not present in the passages.\n\n"
        "Passages:\n{context}\n\nSummary:"
    )
    def __init__(self, llm): self._llm = llm
    def summarize(self, context, max_tokens=200):
        return self._llm.generate(
            self._PROMPT.format(context=context, max_tokens=max_tokens),
            max_tokens=max_tokens,
        )


def _load_error(kind, path_or_id, e):
    return RuntimeError(
        f"raptor: could not load {kind} from {path_or_id!r}. "
        f"Pass an absolute local path or set allow_download=true. Cause: {e}"
    )


class _BGEM3Embedder:
    """BGE-M3 dense (CLS, L2-norm) via transformers."""
    _MAX_LEN = 512
    def __init__(self, path_or_id, max_length=None, allow_download=False):
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as e:
            raise RuntimeError("raptor: requires 'transformers'.") from e
        import torch
        self._torch = torch
        lfo = not allow_download
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(path_or_id, local_files_only=lfo)
            self._model = AutoModel.from_pretrained(path_or_id, local_files_only=lfo)
        except Exception as e:
            raise _load_error("BGE-M3", path_or_id, e) from e
        self._model.eval()
        self._max_length = int(max_length or self._MAX_LEN)
    def _encode(self, texts):
        inputs = self._tokenizer(texts, padding=True, truncation=True,
                                 return_tensors="pt", max_length=self._max_length)
        with self._torch.no_grad():
            out = self._model(**inputs)
        dense = self._torch.nn.functional.normalize(out.last_hidden_state[:, 0], p=2, dim=1)
        return dense.cpu().numpy()
    def encode(self, texts): return self._encode(texts).tolist()
    def encode_query(self, text): return self._encode([text])[0].tolist()


class _SBERTMeanPoolEmbedder:
    """SBERT mean-pool (paper-exact) via transformers."""
    _MAX_LEN = 512
    def __init__(self, path_or_id, max_length=None, allow_download=False):
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as e:
            raise RuntimeError("raptor: requires 'transformers'.") from e
        import torch
        self._torch = torch
        lfo = not allow_download
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(path_or_id, local_files_only=lfo)
            self._model = AutoModel.from_pretrained(path_or_id, local_files_only=lfo)
        except Exception as e:
            raise _load_error("SBERT", path_or_id, e) from e
        self._model.eval()
        self._max_length = int(max_length or self._MAX_LEN)
    def _encode(self, texts):
        inputs = self._tokenizer(texts, padding=True, truncation=True,
                                 return_tensors="pt", max_length=self._max_length)
        with self._torch.no_grad():
            out = self._model(**inputs)
        last = out.last_hidden_state
        mask = inputs["attention_mask"].unsqueeze(-1).expand(last.size()).float()
        pooled = self._torch.sum(last * mask, 1) / self._torch.clamp(mask.sum(1), min=1e-9)
        pooled = self._torch.nn.functional.normalize(pooled, p=2, dim=1)
        return pooled.cpu().numpy()
    def encode(self, texts): return self._encode(texts).tolist()
    def encode_query(self, text): return self._encode([text])[0].tolist()


class _HFSummarizer:
    _PROMPT = _LLMSummarizer._PROMPT
    def __init__(self, path_or_id, allow_download=False):
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
        except ImportError as e:
            raise RuntimeError("raptor: requires 'transformers'.") from e
        lfo = not allow_download
        try:
            tok = AutoTokenizer.from_pretrained(path_or_id, local_files_only=lfo)
            model = AutoModelForCausalLM.from_pretrained(path_or_id, local_files_only=lfo)
        except Exception as e:
            raise _load_error("LLM", path_or_id, e) from e
        self._pipe = pipeline("text-generation", model=model, tokenizer=tok)
    def summarize(self, context, max_tokens=200):
        prompt = self._PROMPT.format(context=context, max_tokens=max_tokens)
        out = self._pipe(prompt, max_new_tokens=max_tokens, do_sample=False)
        text = out[0]["generated_text"]
        return text[len(prompt):].strip() if text.startswith(prompt) else text.strip()


class _OpenAISummarizer:
    _PROMPT = (
        "Please summarize the following text in approximately {max_tokens} "
        "tokens. Preserve all key entities, numbers, and factual claims. "
        "Do not add information not present in the text.\n\n"
        "Text:\n{context}\n\nSummary:"
    )
    def __init__(self, model, api_key=None):
        try:
            import openai
        except ImportError as e:
            raise RuntimeError("raptor: PAPER_EXACT_NATIVE requires 'openai'.") from e
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError(
                "raptor: PAPER_EXACT_NATIVE requires OPENAI_API_KEY. "
                "The adapter will not silently substitute a non-paper LLM."
            )
        self._client = openai.OpenAI(api_key=key)
        self._model = model
    def summarize(self, context, max_tokens=200):
        r = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": self._PROMPT.format(
                context=context, max_tokens=max_tokens)}],
            max_tokens=max_tokens, temperature=0.0,
        )
        return r.choices[0].message.content.strip()


@dataclass
class _RaptorNode:
    node_id: str
    level: int
    text: str
    embedding: list[float]
    parent_doc_id: str
    is_leaf: bool
    member_ids: list[str] = field(default_factory=list)


class RaptorBaseline(BaseBaseline):
    name = "raptor"
    version = "0.2.1"
    status = BaselineStatus.EXECUTABLE
    supports_native = True
    supports_controlled = True
    is_llm_based = True

    def __init__(self, *, config, work_dir, embedder=None, llm=None):
        super().__init__(config=config, work_dir=work_dir, embedder=embedder, llm=llm)
        self._nodes = []
        self._loaded = False
        if self.config.mode == RunMode.CONTROLLED:
            if embedder is None or llm is None:
                raise ValueError("raptor: controlled mode requires both embedder and llm injection")
            self._embedder = _EmbedderWrapper(embedder)
            self._summarizer = _LLMSummarizer(llm)
        else:
            self._embedder, self._summarizer = self._build_native_components()
        extra = self.config.extra or {}
        self.chunk_size = int(extra.get("chunk_size", _DEFAULT_CHUNK_SIZE))
        self.chunk_overlap = int(extra.get("chunk_overlap", _DEFAULT_CHUNK_OVERLAP))
        self.max_levels = int(extra.get("max_levels", _DEFAULT_MAX_LEVELS))
        self.cluster_size = int(extra.get("cluster_size", _DEFAULT_CLUSTER_SIZE))
        self.umap_components = int(extra.get("umap_components", _DEFAULT_UMAP_COMPONENTS))
        self.random_seed = int(extra.get("random_seed", _DEFAULT_RANDOM_SEED))
        self.use_umap = bool(extra.get("use_umap", True))

    def _build_native_components(self):
        extra = self.config.extra or {}
        fidelity = extra.get("native_fidelity", _AUTHOR_SUPPORTED)
        if fidelity == _PAPER_EXACT:
            return self._build_paper_exact(extra)
        if fidelity == _AUTHOR_SUPPORTED:
            return self._build_author_supported(extra)
        raise ValueError(f"raptor: unknown native_fidelity={fidelity!r}")

    def _build_author_supported(self, extra):
        embedder_path = extra.get("embedder_path")
        llm_path = extra.get("llm_path")
        if not embedder_path or not llm_path:
            raise ValueError("raptor: AUTHOR_SUPPORTED requires embedder_path and llm_path")
        allow = bool(extra.get("allow_download", False))
        return (_BGEM3Embedder(embedder_path, allow_download=allow),
                _HFSummarizer(llm_path, allow_download=allow))

    def _build_paper_exact(self, extra):
        # Check API key BEFORE loading any model (avoids accidental downloads).
        api_key = extra.get("openai_api_key") or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "raptor: PAPER_EXACT_NATIVE requires OPENAI_API_KEY. "
                "The adapter will not silently substitute a non-paper LLM."
            )
        sbert = extra.get("sbert_model", _PAPER_EXACT_SBERT_MODEL)
        summ_model = extra.get("openai_summarizer_model", _PAPER_EXACT_SUMMARIZER)
        allow = bool(extra.get("allow_download", False))
        embedder = _SBERTMeanPoolEmbedder(sbert, allow_download=allow)
        summ = _OpenAISummarizer(summ_model, api_key=api_key)
        log.info("raptor: PAPER_EXACT_NATIVE active (sbert=%s, summarizer=%s)", sbert, summ_model)
        return embedder, summ

    def build_index(self, documents):
        if not documents:
            raise ValueError("raptor: build_index requires at least one document")
        self.work_dir.mkdir(parents=True, exist_ok=True)
        leaves = []
        for doc in documents:
            text = doc.text_normalized or doc.text_raw
            for i, chunk in enumerate(self._chunk(text)):
                leaves.append(_RaptorNode(
                    node_id=f"{doc.doc_id}::L0::N{i}", level=0, text=chunk,
                    embedding=[], parent_doc_id=doc.doc_id, is_leaf=True,
                ))
        if not leaves:
            raise ValueError("raptor: no non-empty chunks produced")
        for node, emb in zip(leaves, self._embedder.encode([n.text for n in leaves]), strict=True):
            node.embedding = list(emb)
        all_nodes = list(leaves)
        current = list(leaves)
        for level in range(1, self.max_levels + 1):
            if len(current) <= self.cluster_size:
                break
            clusters = self._cluster(current)
            if not clusters or len(clusters) >= len(current):
                break
            next_level = []
            for ci, idxs in enumerate(clusters):
                members = [current[j] for j in idxs]
                if not members:
                    continue
                summary = members[0].text if len(members) == 1 else \
                    self._summarizer.summarize("\n\n".join(m.text for m in members), max_tokens=200)
                next_level.append(_RaptorNode(
                    node_id=f"summary::L{level}::N{ci}", level=level, text=summary,
                    embedding=[], parent_doc_id=self._dominant_parent(members),
                    is_leaf=False, member_ids=[m.node_id for m in members],
                ))
            if not next_level:
                break
            for node, emb in zip(next_level,
                                 self._embedder.encode([n.text for n in next_level]), strict=True):
                node.embedding = list(emb)
            all_nodes.extend(next_level)
            current = next_level
        self._nodes = all_nodes
        self._persist(all_nodes)
        self._loaded = True
        max_level = max((n.level for n in all_nodes), default=0)
        log.info("raptor: tree built (%d leaves, %d nodes, max_level=%d)",
                 len(leaves), len(all_nodes), max_level)

    def retrieve(self, query, top_k=None):
        if top_k is None:
            top_k = self.config.top_k
        if not self._loaded:
            self._load()
        if not self._nodes:
            return []
        q = np.asarray(self._embedder.encode_query(query), dtype=np.float32)
        mat = np.asarray([n.embedding for n in self._nodes], dtype=np.float32)
        mat = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12)
        q = q / (np.linalg.norm(q) + 1e-12)
        scores = mat @ q
        k = min(int(top_k), len(self._nodes))
        top_idxs = np.argsort(-scores)[:k]
        results = []
        for rank, idx in enumerate(top_idxs, start=1):
            n = self._nodes[int(idx)]
            results.append(RetrievedChunk(
                doc_id=n.parent_doc_id, chunk_id=n.node_id, text=n.text,
                score=float(scores[int(idx)]), rank=rank, parent_doc_id=n.parent_doc_id,
                metadata={"level": n.level, "is_leaf": n.is_leaf},
            ))
        return results

    def _persist(self, nodes):
        with (self.work_dir / "raptor_tree.jsonl").open("w", encoding="utf-8") as f:
            for n in nodes:
                f.write(json.dumps({
                    "node_id": n.node_id, "level": n.level, "text": n.text,
                    "parent_doc_id": n.parent_doc_id, "is_leaf": n.is_leaf,
                    "member_ids": n.member_ids,
                }, ensure_ascii=False) + "\n")
        np.save(self.work_dir / "raptor_embeddings.npy",
                np.asarray([n.embedding for n in nodes], dtype=np.float32))

    def _load(self):
        meta = self.work_dir / "raptor_tree.jsonl"
        emb = self.work_dir / "raptor_embeddings.npy"
        if not meta.exists() or not emb.exists():
            raise RuntimeError(f"raptor: index not found at {self.work_dir}; call build_index() first")
        nodes = []
        with meta.open("r", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                nodes.append(_RaptorNode(
                    node_id=r["node_id"], level=r["level"], text=r["text"], embedding=[],
                    parent_doc_id=r["parent_doc_id"], is_leaf=r["is_leaf"],
                    member_ids=r.get("member_ids", []),
                ))
        arr = np.load(emb)
        for n, e in zip(nodes, arr, strict=True):
            n.embedding = [float(x) for x in e]
        self._nodes = nodes
        self._loaded = True

    def _chunk(self, text):
        words = text.split()
        if not words:
            return []
        if len(words) <= self.chunk_size:
            return [text]
        chunks = []
        step = max(1, self.chunk_size - self.chunk_overlap)
        for start in range(0, len(words), step):
            ch = " ".join(words[start:start + self.chunk_size])
            if ch.strip():
                chunks.append(ch)
            if start + self.chunk_size >= len(words):
                break
        return chunks

    def _cluster(self, nodes):
        if len(nodes) < 2:
            return [list(range(len(nodes)))]
        X = np.asarray([n.embedding for n in nodes], dtype=np.float32)
        use_umap_here = self.use_umap and len(nodes) >= _UMAP_MIN_NODES
        n_components = min(self.umap_components, X.shape[1] - 1, X.shape[0] - 2)
        if use_umap_here and n_components >= 2:
            try:
                import umap
                X_red = umap.UMAP(
                    n_components=n_components, n_neighbors=min(15, X.shape[0] - 1),
                    random_state=self.random_seed, metric="cosine",
                ).fit_transform(X)
            except Exception as e:
                log.warning("raptor: UMAP failed (%s); skipping", e)
                X_red = X
        else:
            X_red = X
        k = max(2, min(len(nodes) // max(1, self.cluster_size), 10))
        if k >= len(nodes):
            k = len(nodes) - 1
        try:
            from sklearn.mixture import GaussianMixture
            gm = GaussianMixture(n_components=k, covariance_type="diag", reg_covar=1e-3,
                                 random_state=self.random_seed, n_init=2, max_iter=100)
            gm.fit(X_red)
            labels = gm.predict(X_red)
        except Exception as e:
            log.warning("raptor: GMM failed (%s); KMeans fallback", e)
            from sklearn.cluster import KMeans
            labels = KMeans(n_clusters=k, random_state=self.random_seed, n_init=2).fit_predict(X_red)
        groups = {}
        for idx, lab in enumerate(labels):
            groups.setdefault(int(lab), []).append(idx)
        return [g for g in groups.values() if g]

    def _dominant_parent(self, members):
        counts = {}
        for m in members:
            counts[m.parent_doc_id] = counts.get(m.parent_doc_id, 0) + 1
        return max(counts.items(), key=lambda kv: kv[1])[0]