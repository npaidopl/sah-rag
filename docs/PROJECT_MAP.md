# PROJECT MAP — SAH-RAG v2

> **Purpose:** Quick reference for humans and LLM sessions.
> Tells you what each top-level folder contains and whether you may touch it.

## Tier system

| Tier | Meaning | You may... |
|---|---|---|
| **1. LIVE** | Active v2 code | Read + edit (carefully) |
| **2. REFERENCE** | v1-derived code, still needed | Read only |
| **3. AUDIT** | Historical v1 records | Read only |
| **4. LEGACY** | v1 scripts, may not work | Ignore |

## Top-level map

### Tier 1 — LIVE (v2 active)

| Path | What it is |
|---|---|
| `sah_rag/` | The SAH-RAG architecture (research contribution) |
| `baselines/base.py` | BaseBaseline contract — **FROZEN** |
| `baselines/longrag/`, `baselines/metadata_rag/` | Reference executable baselines |
| `evaluation/` | Retrieval metrics + runner |
| `orchestration/` | Scheduler + manifest + index |
| `datasets/unified.py` | Unified dataset loader |
| `configs/baselines_registry.yaml` | Registry of 10+ baselines |
| `docs/adapter_contract.md` | Baseline spec — **FROZEN** |
| `docs/report_llm_organizer.md` | LLM operating manual — **FROZEN** |
| `tests/conftest.py` | Pytest shared fixtures |
| `workspace/` | Per-baseline dev areas (gitignored) |
| `pyproject.toml` | Build + deps — **FROZEN** |

### Tier 2 — REFERENCE (v1 code, still needed)

| Path | What it is |
|---|---|
| `baselines/*.py` (flat) | v1 baseline modules (longrag_*, metadata_rag_*) |
| `sah_rag/canonical/` | Canonical dataset integration (loaders, paths, models) |
| `schemas/` | 55 JSON schemas (contracts for artifacts) |

### Tier 3 — AUDIT (historical, do not touch)

| Path | What it is |
|---|---|
| `baseline_audit/` | 75 YAML contracts from v1 experimentation |
| `docs/*.md` (root) | v1 architecture specs (context, not edited) |
| `configs/datasets/`, `configs/execution/`, `configs/pipelines/`, `configs/modes/` | v1 configs |
| `requirements/` | v1 requirements folder (superseded by `pyproject.toml`) |

### Tier 4 — LEGACY (v1, ignore)

| Path | What it is |
|---|---|
| `scripts/` | 200+ v1 scripts (validate_*, run_*, smoke_*, colab_*, ...) |
| `docs/data/`, `docs/pilot/`, `docs/project/`, `docs/reports/` | v1 reports |

### Never touch (gitignored, regenerable)

`workspace/`, `experiments/`, `outputs/`, `_archive/`, `_review/`, `data/raw/`, `data/processed/`, `*.worktrees/`

## Where does a new baseline go?

1. Develop in `workspace/<name>_dev/` — read `docs/report_llm_organizer.md`
2. Graduate to `baselines/<name>/` after tests pass
3. Register in `configs/baselines_registry.yaml` (`status: executable`)

## Where does a new dataset go?

1. Add adapter in `canonical-rag-dataset-suite/` (separate repo)
2. Run its pipeline → outputs land in `data/processed/canonical/<dataset_id>/`
3. Baselines pick it up automatically via `datasets/unified.py`

## What NOT to do

- Do NOT import from `rag_system` (dead v1 name)
- Do NOT modify `docs/adapter_contract.md`, `baselines/base.py`, `docs/report_llm_organizer.md` (frozen)
- Do NOT touch Tier 3/4 files
- Do NOT hardcode `C:\Users\...` paths
- Do NOT commit `workspace/`, `outputs/`, `_archive/`, `_review/`

## Quick reference

| Task | Command |
|---|---|
| Activate venv | `.\.venv\Scripts\Activate.ps1` |
| Run all tests | `python -m pytest tests/ -q` |
| Run one baseline's tests | `python -m pytest workspace/<name>_dev/tests/ -v` |
| List datasets | `python -c "from datasets.unified import list_available_datasets; print(list_available_datasets())"` |
| List baselines | `python -c "import yaml; print([b['name'] for b in yaml.safe_load(open('configs/baselines_registry.yaml'))['baselines']])"` |