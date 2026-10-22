# SAH-RAG Adapter Contract (v2)

**Status: FROZEN**. Any change requires a version bump and migration of all baselines.

Every baseline in `baselines/<name>/` MUST implement the `BaseBaseline` interface
from `baselines/base.py`. This document is the human-readable spec.

---

## 1. Folder structure (per baseline)

```text
baselines/<name>/
├── __init__.py
├── adapter.py              # subclass of BaseBaseline
├── config_native.yaml      # config for native mode
├── config_controlled.yaml  # config for controlled mode
├── README.md               # description, license, deps, status
├── requirements.txt        # extra pip deps (beyond sah-rag[all])
└── tests/
    └── test_adapter.py     # standalone test with mock dataset
```

For **reported-only** baselines (no executable code):

```text
baselines/<name>/
├── __init__.py
├── adapter.py              # subclass of ReportedBaseline
├── reported_metrics.yaml   # paper-reported values, per dataset
└── README.md               # citation, paper URL, license status
```

---

## 2. The `BaseBaseline` interface (frozen)

### Identity attributes (class-level, MUST override)

| Attribute | Type | Meaning |
|---|---|---|
| `name` | `str` | unique identifier, snake_case |
| `version` | `str` | semantic version of the adapter |
| `status` | `BaselineStatus` | EXECUTABLE / REPORTED / SKIPPED |
| `supports_native` | `bool` | can run with its own components |
| `supports_controlled` | `bool` | can run with injected embedder/LLM |
| `is_llm_based` | `bool` | uses an LLM in its pipeline |

### Lifecycle (MUST override)

```python
def build_index(self, documents: list[CanonicalDocument]) -> None:
    """Build retrieval index from the corpus. Persist under work_dir."""

def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
    """Return ranked list of RetrievedChunk (ranks 1..k contiguous)."""
```

### Constructor signature (frozen)

```python
def __init__(
    self,
    *,
    config: BaselineConfig,
    work_dir: Path,
    embedder: Embedder | None = None,
    llm: LLM | None = None,
) -> None:
```

The orchestrator always passes all 4 kwargs. A baseline may ignore
`embedder` (e.g., in native mode with its own embedder) but MUST accept it.

---

## 3. The `RetrievedChunk` schema (frozen)

Every `retrieve()` returns `list[RetrievedChunk]` with these fields:

| Field | Type | Required | Meaning |
|---|---|---|---|
| `doc_id` | `str` | ✅ | parent document identifier |
| `chunk_id` | `str` | ✅ | unique chunk identifier within doc |
| `text` | `str` | ✅ | the chunk's text |
| `score` | `float` | ✅ | similarity/retrieval score (higher = better) |
| `rank` | `int` | ✅ | 1-indexed, contiguous, ascending by score |
| `parent_doc_id` | `str \| None` | ⬜ | for hierarchical baselines |
| `metadata` | `dict` | ⬜ | free-form extras |

The orchestrator uses `doc_id` for Recall@k and `chunk_id` for artifact
provenance. Wrong `doc_id` = wrong metric. Wrong `rank` = tie-breaking fails.

---

## 4. Modes (frozen)

### `native` mode

- Baseline uses its own embedder, LLM, index, hyperparameters.
- Goal: reproduce paper-exact or author-supported configuration.
- Orchestrator passes `embedder=None, llm=None`.
- If baseline needs API keys, use env vars. Document in README.
- If code or license unavailable → move to `reported` status.

### `controlled` mode

- Orchestrator injects `embedder` (BGE-M3) and `llm` (shared model).
- Baseline MUST use the injected components for embedding and generation.
- Baseline MAY keep its own index structure, retrieval logic, rerankers,
  hyperparameters — those are part of its algorithm.
- Goal: prove SAH-RAG's advantage is not due to a newer embedder.
- `is_llm_based=True` baselines MUST call `self.llm.generate()` in pipeline.
  They MAY NOT call external APIs.

### `reported` mode

- No execution. Used when:
  - No public code exists
  - License forbids use
  - Reproducibility impossible within available resources
- `reported_metrics.yaml` contains per-dataset metric values from the paper.
- `adapter.py` subclasses `ReportedBaseline` and implements
  `get_reported_metrics(dataset_id) -> dict[str, float]`.

---

## 5. Configs

### `config_native.yaml` (example)

```yaml
name: longrag
mode: native
top_k: 10
extra:
  embedder: BAAI/bge-m3
  llm: Qwen/Qwen2.5-72B-Instruct
  chunk_size: 4096
  # ... whatever the baseline needs
```

### `config_controlled.yaml` (example)

```yaml
name: longrag
mode: controlled
top_k: 10
extra:
  # embedder + llm are NOT specified — orchestrator injects them
  chunk_size: 4096
```

Every `config_*.yaml` has only these top-level keys: `name`, `mode`,
`top_k`, `extra`. Everything else goes in `extra`.

---

## 6. Hard rules (mandatory)

| # | Rule | Reason |
|---|---|---|
| 1 | NO hardcoded filesystem paths | Reproducibility across machines (laptop + GPU server) |
| 2 | NO hardcoded embedder/LLM in controlled mode | Fairness proof requires injection |
| 3 | NO touching files outside `baselines/<name>/` | Isolation; failing baseline = 1 folder |
| 4 | NO network calls in `tests/` | CI-safe; tests must run offline |
| 5 | NO random seeds without seeding | Determinism for artifact manifests |
| 6 | NO final answer generation in v2 | This phase is IR-only |
| 7 | Artifacts go under `work_dir`, never elsewhere | Orchestrator manages lifecycle |
| 8 | `retrieve()` returns contiguous rank 1..k | Metric computation assumes it |
| 9 | All imports use `sah_rag.*`, `baselines.*`, `datasets.*`, `evaluation.*`, `orchestration.*` | v2 layout; `rag_system` is dead |
| 10 | Python 3.12 only; stdlib + declared deps | Matches project pyproject |

---

## 7. Testing requirements

Every `baselines/<name>/tests/test_adapter.py` MUST:

1. Build a mock corpus of 5–10 tiny documents (inline, not loaded from disk)
2. Call `build_index(corpus)` and assert `work_dir` is populated
3. Call `retrieve(query, top_k=3)` for 2–3 queries
4. Assert:
   - Output is `list[RetrievedChunk]`
   - Length ≤ `top_k`
   - Ranks are 1..len contiguous
   - All `doc_id` in corpus
5. Use `pytest.mark.skipif` for anything requiring Qdrant/GPU/API keys
6. Complete in < 10 seconds on CPU

The orchestrator runs `pytest baselines/<name>/tests/` as gate before
graduation. Failing tests = no merge.

---

## 8. Graduation (workspace → baselines)

A baseline is developed in `workspace/<name>_dev/` and graduates when:

1. All 5 mandatory files present
2. `pytest workspace/<name>_dev/tests/` passes
3. `README.md` documents: description, paper, license, status, deps
4. A representative entry added to `configs/baselines_registry.yaml`
5. Manual review of `adapter.py` (only `BaseBaseline` methods used, no rogue imports)

Graduation command (from project root):

```powershell
Move-Item workspace\<name>_dev baselines\<name>
# then edit configs/baselines_registry.yaml
git add baselines/<name>/ configs/baselines_registry.yaml
git commit -m "feat(baselines): add <name> adapter"
```

---

## 9. What is NOT covered by this contract

- **Generation/evaluation of answers**: v2 is IR-only. `generate()` is removed.
- **Prompt engineering**: no prompts in this contract.
- **LLM-as-judge**: no judge calls.
- **Fine-tuning**: baselines are used as-is (or via injected components).
- **Multi-dataset training**: each baseline is trained/built per-dataset.

If a baseline fundamentally cannot exist without generation, it is
out of scope for v2. Report it as `skipped`.
