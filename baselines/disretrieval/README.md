# DISRetrieval - SAH-RAG v2 Baseline Adapter

**Baseline name:** `disretrieval`
**Version:** 0.2.0
**Status:** `EXECUTABLE` (controlled) / `BLOCKED` (native)
**Supports native:** no (`PAPER_EXACT_NATIVE` blocked, missing backbone)
**Supports controlled:** yes (`CONTROLLED_ADAPTATION`, active)
**LLM-based:** yes (Qwen2.5-1.5B-Instruct at index time)

---

## 1. Paper and reference

- **Paper:** Chen, H., Yang, Y., Li, Y., Zhang, M., & Zhang, M.
  *Beyond Chunking: Discourse-Aware Hierarchical Retrieval for Long
  Document Question Answering.* arXiv:2506.06313 (ACL 2026).
  https://arxiv.org/pdf/2506.06313
- **Official code:** https://github.com/DreamH1gh/ACL2026-DISRetrieval
  Pinned commit: `4f3144042c6eb1e514a0a89ec704d892a66e700c`.
- **Controlled parser:** E-NNRSTParser (Yu et al., ACL 2022).
  Upstream: https://github.com/yunan4nlp/E-NNRSTParser
  License: **Apache-2.0**.
  Released checkpoint (Google Drive, from upstream authors):
  `rst_model/` with `model`, `vocab`, `xlm_rst_tuned/` (fine-tuned
  XLNet backbone, 445 MB).

## 2. Architecture summary

DISRetrieval is a discourse-aware hierarchical retriever. Its
architecture-defining logic is:

1. **Neural RST parsing** of each document into a discourse tree.
2. **Bottom-up LLM summarization** of every internal node.
3. **Node embeddings** over both leaf text and internal summaries.
4. **`rest_topk` traversal**: top-down selection, keeping the top-k
   children at each level.
5. **Leaf expansion**: collect surviving leaves per document.
6. **Evidence assembly**: rank leaves by query similarity; cap at
   `leaf_topk` per document.

Any substitution of step 1 (regex, spaCy, generic constituency parser,
heuristic headings) changes the architecture identity and is forbidden.
The `_SimpleParagraphParser` in `adapter.py` is strictly a **TEST-ONLY**
stub and is never used in reported experiments.

---
## 3. Native mode - `PAPER_EXACT_NATIVE` (BLOCKED)

The native track is **blocked**, not deferred. The official repository
contains parser head weights (`saved_model/100/model`, 15.6 MB, 5 keys:
`mlp_words`, `rescale`, `typeEmb`, `EDULSTM`, `dec`) but **the fine-tuned
XLNet backbone (`xlm_rst_tuned/`) is missing** from the repository.
Without this backbone, the parser head cannot be loaded.

### Blocker reasons

1. **Missing backbone.** `xlm_rst_tuned/` (445 MB fine-tuned XLNet) is
   not distributed. Paper-exact reproduction requires training it from
   scratch on the original data, which is not publicly available.
2. **No LICENSE file** in `DreamH1gh/ACL2026-DISRetrieval`. Default
   copyright applies to the head weights.
3. **Training data mismatch.** The bundled `train.rst` (9.74 MB) is
   CJK-tokenized (multiFieldQA-zh origin), not RST-DT. Reproducing the
   paper configuration requires the exact training corpus, which is not
   published.
4. **No author-supported alternative.** No officially released
   alternative backbone or configuration.

`config_native.yaml` is complete as a reference specification but
carries `native_status: BLOCKED_MISSING_BACKBONE`. The orchestrator
MUST NOT route native-mode execution through this baseline.

## 4. Controlled mode - `CONTROLLED_ADAPTATION` (active)

The controlled track preserves the architecture-defining logic while
substituting only non-architecture-defining components:

| Component | Controlled policy |
|---|---|
| Embedder | **injected** by orchestrator (BGE-M3, CLS-pool, L2-norm) |
| Summarizer LLM | Qwen2.5-1.5B-Instruct (local HF snapshot) |
| RST parser | E-NNRSTParser upstream (Apache-2.0) |
| RST checkpoint | Upstream `rst_model/` (Apache-2.0) |
| Tree construction | Official `RSTProcesser` (from DISRetrieval repo) |
| `rest_topk` traversal | unchanged from the paper |
| Leaf expansion | unchanged from the paper |
| Evidence assembly | unchanged from the paper |
| Evaluator | shared deterministic evaluator |

### 4.1 Pipeline: precompute-then-retrieve

Because parsing and summarization are expensive (neural RST + LLM
inference), the pipeline is split into two phases.

**Phase A - Precompute (offline, one-time per dataset)**

Runs in the dedicated venv
`D:\rag_work\venvs\disretrieval_rst` (torch 2.2.2+cu118,
transformers 4.42.3, Python 3.11). Produces:
D:/rag_work/disretrieval_trees/<dataset>/
manifest.json
doc_<doc_id>.json (tree + embeddings + summaries)

**Phase B - Adapter (online, per query)**

The adapter loads precomputed trees from disk, performs `rest_topk`
traversal, encodes the query with the injected BGE-M3, ranks leaves,
and returns `RetrievedChunk` objects. No parser or LLM runs at query
time.

### 4.2 Verified components (smoke test evidence)

The full controlled pipeline was verified on 2 QASPER documents:

| Metric | doc 1909.12079_4 | doc 1602.04341_11 |
|---|---|---|
| Total nodes | 13 | 41 |
| Leaves | 7 | 21 |
| Internal | 6 | 20 |
| Embeddings (1024-d) | 13 | 41 |
| Empty summaries | 0 | 0 |
| Summaries ? raw text | yes (all) | yes (all) |

Both documents retrieve correctly under multiple queries
(`fine-grained entity typing`, `HABCNN outperforms baselines`,
`Table results performance`) with contiguous rank 1..5 and
cross-document scoring.

---
## 5. Testing

Adapter tests are offline, deterministic, and finish in about one
second on CPU:

```powershell
python -m pytest workspace\disretrieval_dev\tests\ -v
```

The tests exercise:
- identity attributes (`supports_native=False`, `supports_controlled=True`)
- native mode rejection (ValueError)
- controlled on-the-fly build with stub components
- controlled precomputed tree loading (with matching embedder cache)
- missing-tree error path
- rank contiguity (1..k)
- empty-retrieve-before-build
- manifest persistence with `native_fidelity` metadata

**8 tests, all passing** in ~2s.

## 6. Dependencies

**Controlled mode (project venv):**
No extra pip packages. Embedder and LLM are injected by the
orchestrator.

**Precompute (dedicated venv at `D:\rag_work\venvs\disretrieval_rst`):**
- `torch==2.2.2+cu118`
- `transformers==4.42.3`
- `tokenizers==0.19.1`
- `nltk==3.8.1` (with punkt, punkt_tab)
- `numpy==1.24.1`
- `tqdm==4.66.4`
- `setuptools` (for transformers wildcard import)
- `sentencepiece`, `protobuf`, `fire`

**Models (local HF cache):**
- `BAAI/bge-m3` (~2.3 GB, `pytorch_model.bin`)
- `Qwen/Qwen2.5-1.5B-Instruct` (~3.1 GB, `model.safetensors`)
- Upstream E-NNRSTParser checkpoint at
  `D:\rag_work\models\upstream_enNrstparser\extracted_2\rst_model\`

## 7. Known limitations

1. **Native mode is blocked**, not deferred - the backbone does not
   exist publicly and the official repo has no license. Do not route
   native execution through this baseline.
2. **Controlled mode is an adaptation, not a reproduction.** The parser
   head and backbone come from the upstream E-NNRSTParser, not from the
   DISRetrieval paper. This is documented in the config metadata
   (`native_fidelity: CONTROLLED_ADAPTATION`).
3. **Summarization uses Qwen2.5-1.5B**, not the paper's UnifiedQA-3B /
   GPT-class model. This is a controlled-adaptation choice for
   reproducibility without paid API keys.
4. **Precompute must run in the dedicated venv** (torch 2.2.2,
   transformers 4.42.3). The project venv (torch 2.14, transformers
   5.20) is incompatible with the upstream parser code.
5. **Text cleanup**: leaf texts are stored in the tokenized form used
   by the parser (`_NNP`, `<S>`). This is the native format of the
   DISRetrieval pipeline. No de-tokenization is applied because it would
   be an adaptation not covered by the current paper reference.

## 8. Contract compliance checklist

```
[x] adapter.py subclasses BaseBaseline
[x] name, version, status, supports_native=False, supports_controlled=True set
[x] __init__ signature matches contract exactly
[x] build_index() implemented; persists JSON manifest under self.work_dir
[x] retrieve() implemented; returns contiguous RetrievedChunk ranks
[x] config_native.yaml present with BLOCKED metadata
[x] config_controlled.yaml present with CONTROLLED_ADAPTATION metadata
[x] tests/test_adapter.py passes offline, <10s on CPU (8 tests)
[x] README.md documents paper, license, blockers, methodology
[x] No imports from rag_system
[x] No hardcoded absolute paths in adapter
[x] No network calls in tests
[x] No modifications outside workspace/disretrieval_dev/
```

## 9. Graduation request

Graduate the **controlled track only**. The native track stays in
`workspace/` and is not registered as executable - its
`native_status: BLOCKED_MISSING_BACKBONE` flag prevents accidental
execution.

**Registry metadata** (see `configs/baselines_registry.yaml`):
- `native_tag: PAPER_EXACT_NATIVE_BLOCKED`
- `controlled_tag: CONTROLLED_ADAPTATION`
- `native_status: BLOCKED_MISSING_BACKBONE`
- `legal_risk_native: true`