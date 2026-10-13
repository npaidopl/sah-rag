# SAH-RAG

**Structure-Aware Hierarchical Hybrid Retrieval for Long-Context RAG Systems**

SAH-RAG is a deterministic, structure-aware, hierarchical, hybrid retrieval
architecture for Retrieval-Augmented Generation (RAG) in long-context
scenarios. This repository contains its reference implementation, the frozen
experimental contracts used in the accompanying evaluation, and the written
reports of the project.

The architecture is evaluated under a shared fair-comparison protocol against
two external retrieval architectures — **LongRAG** (controlled adaptation)
and **Metadata-RAG** (paper-derived comparator) — on four heterogeneous legal
corpora. SAH-RAG outperforms both on **all eighteen metrics** and **all four
datasets**.

---

## Key Results

Primary metric: **macro-averaged document Recall@10** across four datasets.

| Architecture | Macro Recall@10 |
|---|---|
| LongRAG (controlled adaptation) | 0.3144 |
| Metadata-RAG (paper-derived) | 0.3651 |
| **SAH-RAG (this work)** | **0.4049** |

Relative improvement of SAH-RAG over LongRAG on the primary metric: **+28.79%**.
The advantage grows systematically with decreasing document length, reaching
**+351%** on CLERC (short-document corpus).

### Per-dataset results (document Recall@10)

| Dataset | LongRAG | Metadata-RAG | SAH-RAG |
|---|---|---|---|
| CLERC | 0.0786 | 0.2726 | **0.3546** |
| CUAD | 0.0324 | 0.0294 | **0.0431** |
| GreekBarBench | 0.2606 | 0.2711 | **0.2852** |
| LaRA | 0.8861 | 0.8874 | **0.9368** |
| **Macro** | 0.3144 | 0.3651 | **0.4049** |

---

## Architecture

SAH-RAG organises knowledge in **three levels**:

1. **Semantic parents** — a deterministic structural parser segments each
   document into logical units aligned with headings and article/section
   boundaries. No LLM is used; the parser is fully deterministic and
   byte-reproducible.
2. **Child chunks** — each parent is split into 256-token child chunks with
   32-token overlap, enforcing that child boundaries never cross parent
   boundaries. This preserves document identity during retrieval.
3. **Hybrid representations** — each child is embedded with BGE-M3 to produce
   *dense* (1024-dim, FP16, β„“2-normalised) and *learned-sparse* vectors in a
   single forward pass.

Retrieval fuses the two rankings with Reciprocal Rank Fusion (RRF, k = 60)
and aggregates back to parents and documents via a *best-plus-margin* rule.

### Determinism and reproducibility

Every artifact (parent, chunk, embedding, Qdrant point) has a
content-addressable identity. All serialisations are byte-deterministic.
Every stage is resume-safe via SHA-256 manifests. Two runs with identical
inputs produce byte-identical artifacts.

---

## Repository Structure

```
sah-rag/
β”β”€β”€ README.md                      # This file
β”β”€β”€ LICENSE                        # MIT
β”β”€β”€ CITATION.cff                   # Citation metadata
β”β”€β”€ pyproject.toml                 # Project metadata + dependencies
β”β”€β”€ .gitignore
β”‚
β”β”€β”€ src/rag_system/                # Main source package
β”‚   β”β”€β”€ canonical/                 # Canonical data model
β”‚   β”β”€β”€ parsing/                   # Deterministic structural parser
β”‚   β”β”€β”€ chunking/                  # Hierarchical child chunker
β”‚   β”β”€β”€ embedding/                 # BGE-M3 dense + sparse embedder
β”‚   β”β”€β”€ storage/                   # Qdrant hybrid collections
β”‚   β”β”€β”€ retrieval/                 # Hybrid RRF + aggregation
β”‚   β”β”€β”€ evaluation/                # Metric computation
β”‚   β”β”€β”€ artifacts/                 # Manifest writers & checksums
β”‚   β”β”€β”€ generation/                # (dry-run only; not part of evaluation)
β”‚   β”β”€β”€ orchestration/             # Pipeline planner
β”‚   β””β”€β”€ baselines/                 # LongRAG & Metadata-RAG comparators
β”‚
β”β”€β”€ baseline_audit/                # Frozen contracts & audit artifacts
β”‚   β”β”€β”€ common_*.yaml
β”‚   β”β”€β”€ dataset_*.yaml
β”‚   β”β”€β”€ fair_benchmark_*.yaml
β”‚   β”β”€β”€ longrag_*.yaml
β”‚   β”β”€β”€ metadata_rag_*.yaml
β”‚   β”β”€β”€ my_rag_*.yaml              # historical name; see Naming Note
β”‚   β””β”€β”€ reports/                   # Comparison reports (JSON / MD)
β”‚
β”β”€β”€ docs/                          # Documentation
β”‚   β”β”€β”€ README.md                  # Documentation hub
β”‚   β”β”€β”€ architecture_specification.md
β”‚   β”β”€β”€ fair_benchmark_methodology.md
β”‚   β”β”€β”€ reports/                   # Written reports (LaTeX sources)
β”‚   β”‚   β”β”€β”€ sah_rag_implementation.tex
β”‚   β”‚   β”β”€β”€ sah_rag_optimization.tex
β”‚   β”‚   β”β”€β”€ datasets.tex
β”‚   β”‚   β”β”€β”€ pipelines_status.tex
β”‚   β”‚   β”β”€β”€ longrag_full_track_report.tex
β”‚   β”‚   β”β”€β”€ longrag_vs_sah_rag.tex
β”‚   β”‚   β”β”€β”€ metadata_rag_full_track_report.tex
β”‚   β”‚   β”β”€β”€ three_way_comparison.tex
β”‚   β”‚   β””β”€β”€ archived/              # Reports on non-executed architectures
β”‚   β”β”€β”€ pilot/                     # Infrastructure pilot design docs
β”‚   β””β”€β”€ data/                      # Project PDF
β”‚       β””β”€β”€ REPORT_NIKOLAOS_PAIDOPOULOS.pdf
β”‚
β”β”€β”€ scripts/                       # Execution & validation entry points
β”‚   β”β”€β”€ run_sah_rag_*.py
β”‚   β”β”€β”€ run_longrag_*.py
β”‚   β”β”€β”€ run_metadata_rag_*.py
β”‚   β””β”€β”€ compare_*.py
β”‚
β””β”€β”€ tests/                         # Unit, integration, and synthetic tests
    β”β”€β”€ unit/
    β”β”€β”€ integration/
    β””β”€β”€ fixtures/
```

---

## Quick Start

### Prerequisites

- Python 3.12.10 or newer
- CUDA-capable GPU (NVIDIA T4 or equivalent) for embedding generation
- ~15 GB free disk for intermediate artifacts

### Setup

```bash
python -m venv .venv
source .venv/bin/activate         # on Windows: .venv\\Scripts\\Activate.ps1
pip install -e ".[dev]"
```

### Environment

The canonical dataset root is resolved via environment variable, with a
sensible default:

```bash
# Optional override:
export SAH_RAG_PROJECT_ROOT=/path/to/project-root
export SAH_RAG_CANONICAL_ROOT=/path/to/canonical-datasets
```

### Running the pipeline

```bash
# Full pipeline for SAH-RAG (parsing β†’ chunking β†’ embedding β†’ retrieval β†’ eval)
python scripts/run_sah_rag_m0_full_evaluation.py

# Comparator: LongRAG
python scripts/run_longrag_full_retrieval_batch.py
python scripts/run_longrag_full_evaluation_batch.py

# Comparator: Metadata-RAG
python scripts/run_metadata_rag_evaluation.py

# Three-way comparison
python scripts/compare_metadata_rag_vs_baselines.py
```

All scripts are resume-safe: if a manifest with status `SUCCEEDED` exists,
the script exits without re-running unless `--force` is passed.

---

## Evaluation Protocol

All reported results follow a **retrieval-only** fair-comparison protocol:

| Component | Value |
|---|---|
| Datasets | CLERC, CUAD, GreekBarBench, LaRA |
| Canonical envelope | `documents.jsonl`, `examples.jsonl`, `manifest.json`, `validation.json` |
| Embedding model | `BAAI/bge-m3` @ `5617a9f6...` |
| Embedding precision | FP16 |
| Embedding device | NVIDIA T4 (Google Colab) |
| Evaluation level | canonical document |
| Cutoffs | 1, 5, 10, 20, 50 |
| Primary metric | macro-averaged document Recall@10 |
| Zero-GT policy | excluded, with explicit count |
| Generation / LLM-as-judge | **not used** |

Ground truth is loaded **only inside the evaluator**, which runs as an
independent process. Retrievers receive only `(query_id, query_text)` pairs.

---

## Datasets

The four canonical datasets are hosted in a separate repository:

**[npaidopl/canonical-rag-dataset-suite](https://github.com/npaidopl/canonical-rag-dataset-suite)**

| Dataset | Documents | Queries | Mean doc length | Notes |
|---|---|---|---|---|
| LaRA | 131 | 2 326 | 40 464 tokens | Extreme long-context |
| CLERC | 2 756 | 1 000 | 233 tokens | Short docs, long queries |
| CUAD | 510 | 20 910 | 7 861 tokens | Negative class dominates |
| GreekBarBench | 284 | 284 | 10 162 tokens | Greek-language |

---

## Reports

Written artifacts are available under [`docs/reports/`](docs/reports/):

- **SAH-RAG implementation** — `sah_rag_implementation.tex`
- **SAH-RAG optimization** — `sah_rag_optimization.tex`
- **Datasets** — `datasets.tex`
- **Pipelines status** — `pipelines_status.tex`
- **LongRAG comparator** — `longrag_full_track_report.tex`
- **LongRAG vs SAH-RAG** — `longrag_vs_sah_rag.tex`
- **Metadata-RAG comparator** — `metadata_rag_full_track_report.tex`
- **Three-way comparison** — `three_way_comparison.tex`
- **Non-executed architectures** — `archived/*.tex`

The full Ptoject PDF is in [`docs/data/`](docs/data/).

---

## Citation

If you use this software or reference its results, please cite:

```bibtex
@mastersthesis{paidopoulos2026sahrag,
  author       = {Paidopoulos, Nikolaos},
  title        = {{SAH-RAG}: Structure-Aware Hierarchical Hybrid Retrieval
                  for Long-Context {RAG} Systems},
  school       = {University of Thessaly},
  department   = {Department of Electrical and Computer Engineering},
  year         = {2026},
  type         = {Project},
  url          = {https://github.com/npaidopl/sah-rag}
}
```

A machine-readable citation is also available in [`CITATION.cff`](CITATION.cff).

---

## Naming Note

"My_RAG" was the internal working name of the SAH-RAG architecture during
development. The architecture was renamed to **SAH-RAG** (Structure-Aware
Hierarchical RAG) before publication. All user-facing documentation, source
filenames, and code identifiers use "SAH-RAG". Frozen experimental contracts
in `baseline_audit/` retain the historical name `my_rag_*` to preserve
SHA-256 reproducibility of the associated manifests.

---

## License

This project is released under the **MIT License**. See [`LICENSE`](LICENSE)
for details.

---

## Acknowledgements

This work was carried out at the Department of Electrical and Computer
Engineering, University of Thessaly. The author thanks the examination
committee for their feedback and support.

The implementation builds on the following open-source projects and models:
BGE-M3 (BAAI), Qdrant, Hugging Face Transformers, and PyTorch.
