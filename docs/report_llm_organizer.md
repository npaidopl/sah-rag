# SAH-RAG Baseline Developer — Operating Manual

> **Read this document in full before writing any code.**
> You are one of many parallel LLM sessions. Each session builds exactly ONE
> baseline. Your work must not interfere with the others. Follow every rule.

---

## 1. Your mission (one sentence)

Implement a single baseline adapter that conforms to `baselines/base.py` and
lives entirely under `workspace/<baseline_name>_dev/`, then graduate it to
`baselines/<baseline_name>/` once tests pass.

**You are NOT responsible for:** running evaluations, orchestrating multi-dataset
experiments, comparing results, writing papers, or touching SAH-RAG core code.

**You ARE responsible for:** a working, tested, isolated adapter.

---

## 2. Project context (30 seconds)

SAH-RAG is a research project comparing multiple RAG architectures across
long-context and legal datasets. Each baseline (LongRAG, LightRAG, HippoRAG2,
etc.) is a separate research system. To compare them fairly, every baseline
implements the **same contract** (`BaseBaseline`).

The project uses a **root-package layout** (v2):

```text
SAH-RAG/
├── sah_rag/          ← the SAH-RAG architecture (DO NOT TOUCH)
├── baselines/        ← production baselines (DO NOT TOUCH except your own)
├── evaluation/       ← evaluation code (DO NOT TOUCH)
├── orchestration/    ← scheduler (DO NOT TOUCH)
├── datasets/         ← unified dataset loaders (DO NOT TOUCH)
├── workspace/        ← YOUR WORK AREA
├── configs/          ← registries
├── docs/             ← specs (this file is here)
├── tests/            ← project-wide tests (DO NOT TOUCH)
└── scripts/          ← utility scripts (DO NOT TOUCH)
```

---

## 3. Hard rules (violating any = rejected work)

| # | Rule | Why |
|---|---|---|
| 1 | **Work ONLY inside `workspace/<name>_dev/`** | Isolation; other LLM sessions are running in parallel |
| 2 | **Do not modify `sah_rag/`, `evaluation/`, `orchestration/`, `datasets/`, `tests/`, or any existing baseline** | You will break other work |
| 3 | **Do not touch `.env`, `pyproject.toml`, or `configs/*`** | Environment files are managed elsewhere |
| 4 | **Do not add `rag_system` imports** | That package no longer exists (v1 name) |
| 5 | **No hardcoded absolute paths** | Code must run on laptop AND GPU server |
| 6 | **No network calls in `tests/`** | Tests must run offline |
| 7 | **No random seeds without explicit seeding** | Determinism for reproducibility |
| 8 | **Do not call external LLM APIs** (OpenAI, Anthropic, etc.) | Controlled mode injects LLM; native mode uses local |
| 9 | **Return `list[RetrievedChunk]` from `retrieve()` with ranks 1..k contiguous** | Orchestrator depends on this |
| 10 | **Use Python 3.12 only** | Project requires `>=3.12,<3.13` |

Additional rules:

- Do not add new `pip` dependencies without documenting them in `requirements.txt`
- Do not write to stdout except with `logging` or return values
- Do not commit or push; your supervisor handles that after review

---

## 4. The contract (memorize)

Full spec: `docs/adapter_contract.md`.
Python interface: `baselines/base.py`.

You **MUST** produce a subclass of `BaseBaseline` (or `ReportedBaseline` if no code).

### Identity attributes (override these)

```python
class MyBaseline(BaseBaseline):
    name = "mybaseline"              # snake_case, matches registry
    version = "0.1.0"
    status = BaselineStatus.EXECUTABLE
    supports_native = True
    supports_controlled = True
    is_llm_based = False             # True if it uses an LLM in pipeline
```

### Constructor signature (do not change)

```python
def __init__(
    self,
    *,
    config: BaselineConfig,
    work_dir: Path,
    embedder: Embedder | None = None,
    llm: LLM | None = None,
) -> None:
    super().__init__(config=config, work_dir=work_dir, embedder=embedder, llm=llm)
    # your initialization
```

### Two required methods

```python
def build_index(self, documents: list[CanonicalDocument]) -> None:
    """Build retrieval index. Persist to self.work_dir."""

def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
    """Return ranked chunks. `top_k` defaults to self.config.top_k."""
```

### The `RetrievedChunk` output

```python
RetrievedChunk(
    doc_id="...",           # parent document
    chunk_id="...",         # unique chunk within doc
    text="...",             # the actual text
    score=0.89,             # higher = better
    rank=1,                 # 1-indexed, contiguous, ascending
    parent_doc_id=None,     # optional, for hierarchical
    metadata={},            # optional extras
)
```

Ranks must be exactly `1, 2, 3, ..., k` with no gaps. Rank 1 = best.

---

## 5. What you produce (checklist)

Create `workspace/<name>_dev/` with:

```text
workspace/<name>_dev/
├── __init__.py
├── adapter.py                  # subclass of BaseBaseline
├── config_native.yaml          # native mode config
├── config_controlled.yaml      # controlled mode config
├── README.md                   # description, paper, license, status
├── requirements.txt            # extra deps beyond sah-rag[all]
└── tests/
    └── test_adapter.py         # offline test with mock corpus
```

For **reported-only** baselines (no runnable code):

```text
workspace/<name>_dev/
├── __init__.py
├── adapter.py                  # subclass of ReportedBaseline
├── reported_metrics.yaml       # paper-reported metrics, per dataset
└── README.md                   # citation + license notes
```

---

## 6. Reference implementations (READ THESE)

Two adapters from v1 have been migrated and are your models:

- `baselines/longrag.py` → hierarchical long-unit retrieval
- `baselines/metadata_rag_adapter.py` → metadata-enriched windowing

Before writing your adapter, read both. They show:

- How to use `self.work_dir` for persistence
- How to handle both native and controlled modes via `self.config.mode`
- How to use `self.embedder` and `self.llm` when injected
- What a clean `retrieve()` looks like

If your baseline is graph-based or iterative, note that neither reference
covers those patterns. Follow the contract strictly; add persistence as needed
under `self.work_dir` (e.g., `self.work_dir / "graph.json"`).

---

## 7. Testing (mandatory, offline, <10s on CPU)

Your `tests/test_adapter.py` MUST:

1. Create a mock corpus of 5-10 tiny documents inline (do not load from disk)
2. Call `build_index(corpus)` and assert `work_dir` is populated
3. Call `retrieve(query, top_k=3)` for 2-3 queries
4. Assert:
   - Returns `list[RetrievedChunk]`
   - Length ≤ `top_k`
   - `ranks == [1, 2, ..., len(results)]`
   - Every `doc_id` appears in mock corpus
   - `text` is non-empty
5. No Qdrant, no GPU, no API keys, no network

Minimal skeleton:

```python
import pytest
from pathlib import Path
from sah_rag.canonical.models import CanonicalDocument
from baselines.base import BaselineConfig, RunMode, RetrievedChunk
from workspace.<name>_dev.adapter import MyBaseline


def make_doc(doc_id: str, text: str) -> CanonicalDocument:
    # Replace with the actual CanonicalDocument constructor once verified.
    return CanonicalDocument(doc_id=doc_id, text=text)


def test_build_and_retrieve(tmp_path: Path) -> None:
    docs = [
        make_doc("d1", "The cat sat on the mat."),
        make_doc("d2", "Dogs chase balls in the park."),
        make_doc("d3", "RAG systems retrieve relevant context."),
        make_doc("d4", "Long-context documents need special handling."),
        make_doc("d5", "Legal contracts are structured hierarchically."),
    ]
    cfg = BaselineConfig(name="mybaseline", mode=RunMode.NATIVE, top_k=3)
    baseline = MyBaseline(config=cfg, work_dir=tmp_path / "idx")
    baseline.build_index(docs)
    assert (tmp_path / "idx").exists()

    results = baseline.retrieve("RAG retrieval", top_k=3)
    assert isinstance(results, list)
    assert len(results) <= 3
    assert [r.rank for r in results] == list(range(1, len(results) + 1))
    for r in results:
        assert r.doc_id in {d.doc_id for d in docs}
        assert r.text
```

Verification command:

```powershell
python -m pytest workspace/<name>_dev/tests/ -v
```

Must show all tests passing with 0 failures. Tests > 10s → refactor.

---

## 8. Environment

You are running in the project venv at `.venv/`. Always activate first:

```powershell
.\.venv\Scripts\Activate.ps1
```

Installed packages (from `pyproject.toml`):

- **Core:** pydantic, PyYAML, python-dotenv, platformdirs, typing-extensions
- **Data:** orjson, pandas, pyarrow
- **Storage:** qdrant-client, duckdb
- **Dev:** pytest, pytest-cov, ruff, mypy, jsonschema

If you need an additional dependency, add it to `workspace/<name>_dev/requirements.txt`
and note it in the README. Do not modify `pyproject.toml`.

Do not install heavy deps (torch, transformers, FlagEmbedding) unless your baseline
absolutely requires them; if so, note it prominently in the README so the supervisor
can decide.

---

## 9. Configs

### `config_native.yaml`

```yaml
name: mybaseline
mode: native
top_k: 10
extra:
  # your native hyperparameters
  chunk_size: 512
  # native embedder/LLM if applicable:
  # embedder: BAAI/bge-m3
  # llm: Qwen/Qwen2.5-72B-Instruct
```

### `config_controlled.yaml`

```yaml
name: mybaseline
mode: controlled
top_k: 10
extra:
  # same non-component hyperparameters
  chunk_size: 512
  # NOTE: embedder and llm are INJECTED by orchestrator; do NOT list them here
```

Rules:

- Top-level keys are ONLY: `name`, `mode`, `top_k`, `extra`
- Everything else goes into `extra`
- `mode` must match the filename (`native` or `controlled`)

---

## 10. Graduation (from workspace to production)

When your adapter is complete and tests pass, notify your supervisor. They will:

1. Run `pytest workspace/<name>_dev/tests/ -v` → must pass
2. Review `adapter.py` for contract compliance
3. Move `workspace/<name>_dev` → `baselines/<name>`
4. Update `configs/baselines_registry.yaml` (status: executable)
5. Commit

**You do not graduate yourself.** Do not touch `baselines/`, `configs/`, or `.git`.

---

## 11. What NOT to do (anti-patterns)

| ❌ Don't | ✅ Do |
|---|---|
| Write outside `workspace/<name>_dev/` | Stay isolated |
| Modify `pyproject.toml` | Use `requirements.txt` |
| Hardcode `D:\sah-rag\...` or `/scratch/...` | Use `self.work_dir`, `RAG_CANONICAL_ROOT` |
| Import from `rag_system` | Import from `sah_rag`, `baselines`, `datasets` |
| Call `openai.ChatCompletion` or similar | Use `self.llm.generate()` in controlled mode |
| Return chunks with rank gaps | Always `[1, 2, ..., k]` |
| Use `time.sleep` in tests | Use mocks |
| Silently swallow exceptions in `retrieve()` | Return empty list OR raise clearly |
| Add pytest markers without registering them | Add to `tests/conftest.py` via supervisor |
| Push code to git | Leave for supervisor |

---

## 12. Definition of Done — checklist

Before reporting "done", verify ALL of the following:

```text
□ workspace/<name>_dev/adapter.py subclasses BaseBaseline (or ReportedBaseline)
□ name, version, status, supports_native, supports_controlled, is_llm_based set
□ __init__ signature matches contract exactly
□ build_index() implemented; persists under self.work_dir
□ retrieve() implemented; returns contiguous RetrievedChunk ranks
□ Both config_native.yaml and config_controlled.yaml present with correct schema
□ tests/test_adapter.py passes with pytest (offline, <10s)
□ README.md documents: description, paper, license, status, deps
□ requirements.txt present (even if empty)
□ No imports from rag_system
□ No hardcoded absolute paths
□ No network calls in tests
□ No modifications outside workspace/<name>_dev/
```

---

## 13. What to report back to supervisor

When done, output a structured summary:

```text
BASELINE: <name>
STATUS: complete / blocked
FILES CREATED:
  - workspace/<name>_dev/adapter.py
  - workspace/<name>_dev/config_native.yaml
  - workspace/<name>_dev/config_controlled.yaml
  - workspace/<name>_dev/README.md
  - workspace/<name>_dev/requirements.txt
  - workspace/<name>_dev/tests/test_adapter.py

TEST RESULT:
  <pytest output, last 5 lines>

LICENSE:
  <e.g., MIT / Apache-2.0 / None — paper only>

EXTRA DEPENDENCIES:
  <list or "none">

KNOWN LIMITATIONS:
  <e.g., "no native mode, controlled only because original code uses OpenAI API">

ANY DEVIATIONS FROM CONTRACT:
  <e.g., "none" or explanation>

ASKING FOR GRADUATION: yes / no
```

If blocked: state the blocker (missing code, license, etc.) and **stop**. Do not improvise.

---

## 14. Quick start (5 commands)

```powershell
# 1. Activate venv
.\.venv\Scripts\Activate.ps1

# 2. Create your workspace folder
New-Item -ItemType Directory -Force -Path workspace\<name>_dev\tests

# 3. Read the contract and reference implementations
Get-Content docs\adapter_contract.md
Get-Content baselines\base.py
Get-Content baselines\longrag.py

# 4. Write your adapter, configs, tests, README

# 5. Verify
python -m pytest workspace\<name>_dev\tests\ -v
```

Then report back using the template in Section 13.

---

**Final reminder:** You are building ONE baseline. Isolation is the highest
priority. If you are unsure about anything — **stop and ask your supervisor.
Do not guess.**
