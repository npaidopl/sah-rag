# RAPTOR Baseline (SAH-RAG v2)

**Status:** `EXECUTABLE` (workspace — awaiting graduation)
**Adapter:** `baselines/raptor/adapter.py` → class `RaptorBaseline`
**Version:** 0.1.0

---

## 1. What this baseline is

RAPTOR (**R**ecursive **A**bstractive **P**rocessing for **T**ree-**O**rganized
**R**etrieval) is a retrieval architecture that builds a **multi-level tree**
over a corpus: leaves are text chunks; internal nodes are LLM-generated
abstractive summaries of clusters of their children. Retrieval operates over
the **collapsed tree**, i.e. all nodes at all levels compete in a single
similarity search.

**Reference**

- Paper: Sarthi, P., Abdullah, S., Tuli, A., Khanna, S., Goldie, A., Manning, C. D.
  *RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval*.
  ICLR 2024. arXiv:[2401.18059](https://arxiv.org/abs/2401.18059)
- Official code: <https://github.com/parthsarthi03/raptor>
- License of reference implementation: **MIT**

## 2. Fidelity level

This adapter is **AUTHOR_SUPPORTED_NATIVE**, not PAPER_EXACT_NATIVE.

- **Architecture-defining logic is preserved**: recursive GMM+UMAP clustering,
  LLM summarization at every level, collapsed-tree retrieval over leaves and
  summaries.
- **Pluggable components** are wired through the same abstract interfaces the
  official repository exposes (`BaseEmbeddingModel`, `BaseSummarizationModel`).
- **Why not paper-exact**: the paper uses `gpt-3.5-turbo` (summarizer) and
  `gpt-4` (QA). Both are behind paid APIs whose historical snapshots are
  time-limited. Reproducing them verbatim is therefore not achievable in this
  project's budget. This is a **known and documented** deviation; it is not a
  modification of the architecture.
- **What is NOT changed** to earn "native": the tree is still *recursive* and
  *abstractive*. A flat tree or a leaf-only retriever is classified
  `NOT_RAPTOR_COMPLETE` and the adapter refuses to emit such an index
  (see §7).

## 3. Modes

| Mode | When | What is loaded |
|---|---|---|
| `native` | Reproducing RAPTOR as designed | `embedder_path` and `llm_path` from `config_native.yaml` (`extra`) |
| `controlled` | Track A: fair comparison under shared components | `embedder` (BGE-M3) and `llm` (shared summarizer) injected by the orchestrator |

In **native** mode the adapter loads its own `SentenceTransformer` (default
`BAAI/bge-m3`) and its own HF causal LM (default `Qwen/Qwen2.5-1.5B-Instruct`).
Both paths are overridable in `config_native.yaml`.

In **controlled** mode the adapter refuses to start unless both `embedder` and
`llm` are injected — this is enforced by the frozen `baselines/base.py`
contract (`is_llm_based = True`).
## 4. Config schema

Both configs use only four top-level keys — `name`, `mode`, `top_k`, `extra` —
as mandated by `docs/adapter_contract.md` §5.

### config_native.yaml

~~~yaml
name: raptor
mode: native
top_k: 10
extra:
  embedder_path: BAAI/bge-m3
  llm_path: Qwen/Qwen2.5-1.5B-Instruct
  chunk_size: 100
  chunk_overlap: 20
  max_levels: 4
  cluster_size: 8
  umap_components: 32
  random_seed: 42
~~~

### config_controlled.yaml

~~~yaml
name: raptor
mode: controlled
top_k: 10
extra:
  chunk_size: 100
  chunk_overlap: 20
  max_levels: 4
  cluster_size: 8
  umap_components: 32
  random_seed: 42
~~~

`embedder_path` and `llm_path` are deliberately absent from
`config_controlled.yaml` — the orchestrator injects the components.
## 5. Algorithm hyperparameters

| Param | Default | Meaning |
|---|---|---|
| `chunk_size` | 100 | tokens (word-level approximation) per leaf chunk |
| `chunk_overlap` | 20 | word overlap between consecutive chunks |
| `max_levels` | 4 | maximum depth of the summary tree (paper uses 3–5) |
| `cluster_size` | 8 | target cluster size — drives the number of clusters per level |
| `umap_components` | 32 | dimensionality of UMAP-reduced space before GMM |
| `random_seed` | 42 | seeds UMAP, GMM, KMeans fallback |

The implementation adds two pragmatic safeguards that do not alter the
architecture:

1. UMAP is skipped for tiny inputs (fewer than ~3 nodes, or fewer than 2
   usable components). This avoids degenerate embeddings on the mock corpora
   used in tests. On real datasets UMAP always runs.
2. GMM falls back to KMeans if Gaussian Mixture fails to converge. This is
   an implementation robustness guard, not an architectural change; the paper
   prefers GMM and GMM is the default.

## 6. Persistence layout

`build_index()` writes everything under `self.work_dir`:

~~~text
<work_dir>/
├── raptor_tree.jsonl        # one JSON per node: node_id, level, text,
│                            #   parent_doc_id, is_leaf, member_ids
└── raptor_embeddings.npy    # float32 matrix, rows aligned with JSONL order
~~~

`retrieve()` loads these lazily on first call, so a fresh process can retrieve
without re-building the tree (provided `work_dir` was populated).

## 7. The NOT_RAPTOR_COMPLETE guard

The adapter is designed so that a broken tree cannot be silently reported as
RAPTOR:

- `build_index()` raises if the tree ends up with a single level and the
  corpus is large enough to require clustering. Small mock corpora in tests
  are exempted by design.
- In controlled mode the tree is mandatory: the adapter refuses to produce a
  flat retriever.
- If a future refactor ever collapses the tree, it will fail the tests in
  `tests/test_adapter.py` that assert multi-level metadata on retrieved nodes.
## 8. Dependencies

`requirements.txt` in this folder lists:

- `numpy`, `scikit-learn`, `umap-learn` — required for clustering (tests and
  all modes).
- `sentence-transformers`, `transformers`, `accelerate`, `torch` — required
  only for native mode; commented out in `requirements.txt` and enabled on the
  GPU server.

Installation:

~~~powershell
.\.venv\Scripts\Activate.ps1
pip install -r baselines/raptor\requirements.txt
~~~

## 9. Tests

~~~powershell
python -m pytest baselines/raptor\tests\ -v
~~~

Expected: 5 passed, 1 skipped. The skipped test is the GPU-only native
end-to-end smoke test, enabled on the training server.

| Test | What it verifies |
|---|---|
| `test_controlled_build_and_retrieve` | Full pipeline: 10 docs → tree with summaries → top-k retrieval → contiguous ranks, valid `doc_id`, non-empty text |
| `test_controlled_retrieve_top_k_defaults` | `top_k=None` falls back to `config.top_k` |
| `test_native_requires_paths` | Native mode fails fast without `embedder_path` / `llm_path` |
| `test_controlled_requires_embedder` | Base-contract guard: controlled mode without embedder is rejected |
| `test_controlled_requires_llm_when_llm_based` | Base-contract guard: LLM-based baseline without injected LLM is rejected |

All tests run offline, on CPU, in <30s on the development laptop (dominated
by UMAP import time). They use a deterministic `_MockEmbedder` and `_MockLLM`;
no network, no API keys, no GPU.

## 10. Known limitations

1. Not paper-exact. See §2. `gpt-3.5-turbo` / `gpt-4` are not used; a local
   Qwen model (native) or the shared summarizer (controlled) stands in. This
   is documented and deliberate, not a silent substitution.
2. Word-level chunking. The paper uses tiktoken-based token chunks. This
   adapter uses word-level chunks with `chunk_size` in words as a close
   approximation, to avoid a hard `tiktoken` dependency in tests. The native
   GPU path can be switched to tiktoken by changing `_chunk()`; it is a
   local, non-architectural change.
3. Collapsed-tree only. The paper shows that collapsed-tree retrieval
   outperforms tree-traversal retrieval by 2-4.5 points. This adapter
   implements only the collapsed variant. Tree-traversal is out of scope for
   v2.
4. No generation. SAH-RAG v2 is IR-only (`docs/adapter_contract.md` §6, §9).
   The RAPTOR reader (`gpt-4` in the paper) is not exercised.
5. UMAP cost. On large corpora UMAP fitting dominates indexing time. This is
   expected and matches the paper.
## 11. Graduation checklist

Before moving `baselines/raptor/` → `baselines/raptor/`:

- [ ] `python -m pytest baselines/raptor\tests\ -v` → 5 passed, 1 skipped
- [ ] `configs/baselines_registry.yaml` gets a `raptor` entry with
      `status: executable`
- [ ] Supervisor reviews `adapter.py` for rogue imports (must be only
      `baselines.*`, `sah_rag.*`, stdlib, `numpy`, `sklearn`, `umap`)
- [ ] `README.md` and `requirements.txt` moved verbatim
- [ ] No changes outside `baselines/raptor/`

## 12. Citation

~~~bibtex
@inproceedings{sarthi2024raptor,
  title     = {RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval},
  author    = {Sarthi, Parth and Abdullah, Salman and Tuli, Aditi and
               Khanna, Shubh and Goldie, Anna and Manning, Christopher D.},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2024},
  url       = {https://arxiv.org/abs/2401.18059}
}
~~~

## 13. License

The reference RAPTOR implementation is released under the MIT License.
This adapter is an independent reimplementation of the same algorithm and
inherits no license restrictions from the reference code. It ships under the
same license as the surrounding SAH-RAG project (see the repository root).