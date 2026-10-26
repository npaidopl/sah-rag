# RAPTOR Baseline (SAH-RAG v2)

**Status:** `EXECUTABLE` | **Version:** 0.2.1
**Adapter:** `baselines/raptor/adapter.py` → `RaptorBaseline`
**Tests:** 6 passed, 1 skipped (4.2s, offline)
**Smoke test:** native E2E verified with real BGE-M3 + Qwen2.5-1.5B (CPU)

---

## 1. What this baseline is

RAPTOR (Recursive Abstractive Processing for Tree-Organized Retrieval)
builds a multi-level tree over a corpus: leaves are text chunks; internal
nodes are LLM-generated abstractive summaries of clusters of their children.
Retrieval operates over the **collapsed tree** — all nodes at all levels
compete in a single similarity search.

**Reference**

- Paper: Sarthi et al., *RAPTOR: Recursive Abstractive Processing for
  Tree-Organized Retrieval*, ICLR 2024. arXiv:[2401.18059](https://arxiv.org/abs/2401.18059)
- Official code: <https://github.com/parthsarthi03/raptor> (MIT License)

## 2. Fidelity levels

Two native fidelities are implemented in the adapter. The choice is set
via `config.extra.native_fidelity`.

### `AUTHOR_SUPPORTED_NATIVE` (default, executable offline)

- The recursive abstractive tree is preserved exactly.
- Components are plugged through the same abstract interfaces the official
  repository exposes (`BaseEmbeddingModel`, `BaseSummarizationModel`).
- Default stack: **BGE-M3** embedder + a **local HuggingFace causal LM**
  (Qwen2.5-1.5B-Instruct by default). No API keys.
- This is what runs in the SAH-RAG v2 IR-only phase.

### `PAPER_EXACT_NATIVE` (fully implemented, requires `OPENAI_API_KEY`)

- Paper-exact component choices:
  - Embedder: `sentence-transformers/multi-qa-mpnet-base-cos-v1` (SBERT)
  - Summarizer: `gpt-3.5-turbo`
  - QA: `gpt-4` (not exercised in v2 IR-only)
- The adapter **refuses to run without `OPENAI_API_KEY`** — no silent
  fallback to a substitute LLM. Once the key is available, this mode is a
  direct paper-exact reproduction.
- Encodes SBERT via transformers + mean-pooling (numerically equivalent to
  SBERT defaults; avoids a hard sentence-transformers dependency).

### Why `PAPER_EXACT_NATIVE` is deferred (not skipped)

The historical `gpt-3.5-turbo` snapshot used by the paper is time-limited
and paid. The adapter fully implements it; execution is blocked only on
the API key. Documenting this explicitly makes the difference auditable:
a reviewer can verify that paper-exact is *implemented*, and only the
*execution* is deferred for cost reasons.

## 3. Modes

| Mode | When | Components |
|---|---|---|
| `native` | Reproducing RAPTOR as designed | Loaded per `native_fidelity` (see §2) |
| `controlled` | Track A: fair comparison under shared components | `embedder` (BGE-M3) + `llm` (shared summarizer) injected by orchestrator |

Both modes preserve the recursive tree. A flat tree is
`NOT_RAPTOR_COMPLETE` and cannot be reported as RAPTOR (enforced by tests).

## 4. Local model paths (this machine)

Verified absolute paths (avoid HF cache resolution, prevent accidental
downloads):

- **BGE-M3**:
  `D:/rag_work/hf_cache/models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181`
- **Qwen2.5-1.5B-Instruct**:
  `D:/rag_work/hf_cache/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306`

**Do not use** `D:/rag_work/hf_cache/hub/models--*/` — those subfolders
are **incomplete** (config + tokenizer only, no weights).

## 5. Config schema

Top-level keys: `name`, `mode`, `top_k`, `extra`. Everything else goes
into `extra`.

### `config_native.yaml` (AUTHOR_SUPPORTED_NATIVE)

~~~yaml
name: raptor
mode: native
top_k: 10
extra:
  native_fidelity: AUTHOR_SUPPORTED_NATIVE
  allow_download: false
  embedder_path: D:/rag_work/hf_cache/models--BAAI--bge-m3/snapshots/<hash>
  llm_path: D:/rag_work/hf_cache/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/<hash>
  chunk_size: 100
  chunk_overlap: 20
  max_levels: 4
  cluster_size: 8
  umap_components: 32
  random_seed: 42
~~~

To switch to `PAPER_EXACT_NATIVE`, set `native_fidelity: PAPER_EXACT_NATIVE`
and ensure `OPENAI_API_KEY` is set in the environment (no other change
needed; the adapter reads `sbert_model` / `openai_summarizer_model` from
`extra` with paper-correct defaults).

### `config_controlled.yaml`

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

`embedder_path` / `llm_path` are deliberately absent — the orchestrator
injects the components.
## 6. Algorithm hyperparameters

| Param | Default | Meaning |
|---|---|---|
| `chunk_size` | 100 | words per leaf chunk (paper uses ~100 tokens) |
| `chunk_overlap` | 20 | word overlap between consecutive chunks |
| `max_levels` | 4 | maximum depth of the summary tree (paper: 3–5) |
| `cluster_size` | 8 | target cluster size; drives clusters-per-level |
| `umap_components` | 32 | UMAP reduction dimension before GMM |
| `random_seed` | 42 | seeds UMAP, GMM, KMeans fallback |
| `allow_download` | false | if false, `local_files_only=True` on model loads |
| `use_umap` | true | skip UMAP when `len(nodes) < 30` (JIT warmup guard) |

Two pragmatic safeguards (no architectural change):

1. **UMAP skipped** for tiny inputs (< 30 nodes). Avoids numba JIT's ~15s
   warm-up on mock corpora. On real datasets, UMAP always runs.
2. **GMM → KMeans fallback** if Gaussian Mixture fails to converge.
   Paper's GMM is default.

## 7. Persistence

`build_index()` writes under `self.work_dir`:

~~~text
<work_dir>/
├── raptor_tree.jsonl        # one JSON per node (node_id, level, text,
│                            #   parent_doc_id, is_leaf, member_ids)
└── raptor_embeddings.npy    # float32 matrix, rows aligned with JSONL
~~~

`retrieve()` loads lazily on first call. A fresh process can retrieve
without re-building.

## 8. The NOT_RAPTOR_COMPLETE guard

- `build_index()` raises if the corpus is large enough to require clustering
  but the tree stays at level 0.
- Controlled mode refuses to flatten the tree.
- Tests assert multi-level metadata on retrieved nodes.

## 9. Embedders: why transformers, not sentence-transformers

The project has a local package named `datasets` (FROZEN) that shadows
HuggingFace's `datasets` when the project root is on `sys.path`. This
breaks `sentence-transformers` (which imports `datasets`). Our embedders
use `transformers.AutoModel` directly:

- `_BGEM3Embedder`: CLS token, L2-normalized → BGE-M3 dense spec
- `_SBERTMeanPoolEmbedder`: mean pooling over attention mask, L2-normalized
  → SBERT default pooling for the paper's mpnet model

Numerically equivalent to the standard libraries.

## 10. Tests

~~~powershell
python -m pytest baselines\raptor\tests\ -v --durations=5
~~~

Expected: **6 passed, 1 skipped in <10s** (offline, CPU).

| Test | Verifies |
|---|---|
| `test_controlled_build_and_retrieve` | Full pipeline: 10 docs → tree → retrieval → contiguous ranks |
| `test_controlled_retrieve_top_k_defaults` | `top_k=None` uses `config.top_k` |
| `test_native_requires_paths` | Native mode without paths fails fast |
| `test_controlled_requires_embedder` | Base contract guard (no embedder → reject) |
| `test_controlled_requires_llm_when_llm_based` | Base contract guard (no llm → reject) |
| `test_paper_exact_requires_openai_key` | PAPER_EXACT without key raises in <1s (no model load) |
| `test_native_end_to_end` (skipped) | Real-model E2E; enabled manually on GPU server |

## 11. Smoke test

~~~powershell
python baselines\raptor\scripts\smoke_test.py --mode dry
python baselines\raptor\scripts\smoke_test.py --mode native --embedder-path <BGE_PATH> --llm-path <QWEN_PATH>
python baselines\raptor\scripts\smoke_test.py --mode controlled --embedder-path <BGE_PATH> --llm-path <QWEN_PATH>
python baselines\raptor\scripts\smoke_test.py --mode paper    # needs OPENAI_API_KEY
~~~

**Verified result on this machine (CPU only, ~102s native):**
[4] Tree: L0=10, L1=3
[5] 'capital of France' → rank=1 Paris doc (score 0.7070)
[5] 'RAG systems' → rank=1 ML/RAG doc (score 0.6092)
[OK] Smoke test passed

Semantically correct retrieval confirms the entire pipeline (chunk → embed
→ cluster → summarize → re-embed → collapsed retrieval) works end-to-end
with real models.

## 12. Known limitations

1. **Word-level chunking.** Paper uses tiktoken; this uses `chunk_size` in
   words. Close approximation, avoids a hard `tiktoken` dependency. The
   native GPU path can switch to tiktoken by changing `_chunk()` — a local,
   non-architectural change.
2. **Collapsed-tree only.** The paper's preferred variant (Sec 4.2: +2–4.5 pp
   vs tree-traversal). Tree-traversal is out of scope.
3. **No generation.** SAH-RAG v2 is IR-only (`docs/adapter_contract.md` §6/§9).
   The reader (`gpt-4`) is not exercised.
4. **PAPER_EXACT_NATIVE not executed.** Implementation complete; execution
   deferred on `OPENAI_API_KEY` (paid). Adapter raises without key, so
   unattended runs cannot silently degrade.
5. **CPU-only locally.** Native smoke test ~102s on this laptop. GPU server
   expected <10s.

## 13. Graduation status

| Item | Status |
|---|---|
| Path | `baselines/raptor/` |
| Registry | `status: executable`, `native_tag: AUTHOR_SUPPORTED_NATIVE` |
| Unit tests | 6 passed, 1 skipped (4.2s) |
| Native E2E smoke | ✅ verified with real models |
| Controlled smoke | ✅ verified with real models |
| Commit | `<to be filled>` |
| Push | pending supervisor |

## 14. Citation

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

## 15. License

Reference RAPTOR implementation is MIT-licensed. This adapter is an
independent reimplementation and ships under the SAH-RAG project license.