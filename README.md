# SAH-RAG

**Structure-Aware Hierarchical Hybrid Retrieval for Long-Context RAG Systems**

Reference implementation and evaluation framework for the SAH-RAG architecture,
developed as a Diploma Thesis at the Department of Electrical and Computer
Engineering, University of Thessaly.

> **Author:** Nikolaos Paidopoulos  
> **Supervisor:** Prof. Emmanouil Vavalis  
> **Institution:** University of Thessaly, School of Engineering  
> **Date:** July 2026

---

## Overview

SAH-RAG is a deterministic, structure-aware, hierarchical, hybrid retrieval
architecture for Retrieval-Augmented Generation (RAG) in long-context scenarios.
It organises knowledge in three levels:

1. **Documents** are parsed by a deterministic structural parser that produces
   *semantic parents* aligned with logical document units (articles, sections,
   headings).
2. **Parents** are split into *child chunks* that never cross parent boundaries,
   keeping document identity intact during retrieval.
3. **Child chunks** are embedded with BGE-M3 to produce *dense* and *learned-sparse*
   representations in a single forward pass. The two rankings are fused with
   Reciprocal Rank Fusion (RRF, k = 60) and aggregated back to parents and
   documents via a *best-plus-margin* rule.

The implementation emphasises **determinism and reproducibility**: every artifact
(parent, chunk, embedding, Qdrant point) has a content-addressable identity, all
serialisations are byte-deterministic, and every stage is resume-safe via SHA-256
manifests.

---

## Key Results

Evaluated on four heterogeneous legal corpora under a shared fair-comparison
protocol (same embedder, same queries, same ground truth, same evaluator):

| Architecture | Macro document Recall@10 |
|---|---|
| LongRAG (controlled adaptation) | 0.3144 |
| Metadata-RAG (paper-derived) | 0.3651 |
| **SAH-RAG (this work)** | **0.4049** |

SAH-RAG outperforms both comparators on **all eighteen metrics** and **all four
datasets**. The relative improvement over LongRAG on the primary metric is
**+28.79%**. The advantage grows systematically with decreasing document length,
reaching **+351%** on CLERC (short-document corpus).

### Per-dataset results (document Recall@10)

| Dataset | LongRAG | Metadata-RAG | SAH-RAG |
|---|---|---|---|
| CLERC | 0.0786 | 0.2726 | **0.3546** |
| CUAD | 0.0324 | 0.0294 | **0.0431** |
| GreekBarBench | 0.2606 | 0.2711 | **0.2852** |
| LaRA | 0.8861 | 0.8874 | **0.9368** |
| **Macro** | 0.3144 | 0.3651 | **0.4049** |

---

## Key Contributions

1. **Architecture.** A novel retrieval architecture combining structural
   awareness, hierarchical organisation, and hybrid (dense + learned-sparse)
   retrieval signal, with full determinism and reproducibility guarantees.
2. **Fair-comparison protocol.** A shared evaluation protocol that isolates
   architectural variables (granularity, retrieval signal, aggregation rule)
   while holding embedder, queries, ground truth, and evaluator constant.
3. **Extensive empirical evaluation.** Parameter-response curves and comparative
   performance measurements across four heterogeneous corpora, reported
   transparently, including negative findings (H1 chunk-size plateau not observed).
4. **Reusable infrastructure.** A deterministic, resume-safe pipeline with
   content-addressable identities and SHA-256 manifests, designed for extension
   to new corpora and new research scenarios.

---

## Repository Structure

```
sah-rag/
├── README.md                      # This file
├── LICENSE                        # MIT
├── CITATION.cff                   # Citation metadata
├── pyproject.toml                 # Project metadata + dependencies
├── .gitignore
│
├── baseline_audit/                # Frozen contracts & audit artifacts
│   ├── common_*.yaml
│   ├── dataset_*.yaml
│   ├── longrag_*.yaml
│   ├── metadata_rag_*.yaml
│   ├── my_rag_*.yaml
│   ├── external_canonical_suite_*.yaml
│   └── reports/                   # Comparison reports (JSON / MD)
│
├── docs/                          # Documentation & written reports
│   ├── README.md
│   ├── fair_benchmark_methodology.md
│   ├── architecture.md
│   └── data/                      # Thesis & comparison reports
│       ├── REPORT_NIKOLAOS_PAIDOPOULOS.pdf
│       ├── metadata_rag_full_track_report.tex
│       ├── three_way_comparison.tex
│       └── README.md
│
├── src/rag_system/                # Main source package
│   ├── canonical/                 # Canonical data model
│   ├── parsing/                   # Deterministic structural parser
│   ├── chunking/                  # Hierarchical child chunker
│   ├── embedding/                 # BGE-M3 dense + sparse embedder
│   ├── storage/                   # Qdrant hybrid collections
│   ├── retrieval/                 # Hybrid RRF + aggregation
│   ├── evaluation/                # Metric computation
│   ├── artifacts/                 # Manifest writers & checksums
│   └── baselines/                 # LongRAG & Metadata-RAG adapters
│
├── scripts/                       # Execution & validation scripts
│   ├── run_my_rag_*.py
│   ├── run_longrag_*.py
│   ├── run_metadata_rag_*.py
│   └── compare_*.py
│
└── tests/                         # Unit, integration, and synthetic tests
    ├── unit/
    ├── integration/
    └── fixtures/
```

---

## Architecture Summary

### Offline stage (knowledge preparation)

1. **Structural parsing** — `DeterministicStructuralParser` segments the
   canonical text into paragraphs, detects headings by three cascading rules
   (named, numbered, uppercase), and groups paragraphs into semantic parents.
2. **Hierarchical chunking** — `HierarchicalChildChunker` splits each parent
   into 256-token child chunks with 32-token overlap, enforcing that child
   boundaries never cross parent boundaries.
3. **Embedding** — BGE-M3 (`BAAI/bge-m3` @ pinned revision) produces dense
   (1024-dim, FP16, ℓ2-normalised) and learned-sparse representations in one
   forward pass.
4. **Storage** — Qdrant collections with named vectors (`dense` + `sparse`)
   and deterministic point IDs via `uuid5(NAMESPACE_URL, "rag-chunk:" + chunk_id)`.

### Online stage (retrieval)

1. Query is embedded with the same pinned BGE-M3 revision.
2. Dense top-K and sparse top-K candidate sets are retrieved independently
   (K = 200 by default).
3. Ranked lists are fused with Reciprocal Rank Fusion (k = 60).
4. Fused child chunks are aggregated to parents via
   `score(P) = max(score(m)) + w * Σ score(m\\{m*})` with `w = 0.05`.
5. Parents are resolved to canonical documents via
   `score(D) = max(score(P))`.

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
# Full offline pipeline (parsing → chunking → embedding → storage)
python scripts/run_my_rag_m0_full_evaluation.py

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

Written artifacts are available under [`docs/data/`](docs/data/):

- **`REPORT_NIKOLAOS_PAIDOPOULOS.pdf`** — full Diploma Thesis (Greek, with English abstract).
- **`metadata_rag_full_track_report.tex`** — Metadata-RAG comparator report.
- **`three_way_comparison.tex`** — SAH-RAG vs LongRAG vs Metadata-RAG.

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
  type         = {Diploma Thesis},
  url          = {https://github.com/npaidopl/sah-rag}
}
```

A machine-readable citation is also available in [`CITATION.cff`](CITATION.cff).

---

## License

This project is released under the **MIT License**. See [`LICENSE`](LICENSE)
for details.

---

## Acknowledgements

Supervisor: **Prof. Emmanouil Vavalis**.

Special thanks to the examination committee members **Prof. George Thanos**
and **Prof. Elias Houstis** for their feedback and support.

This work used the following open-source projects and models:
BGE-M3 (BAAI), Qdrant, Hugging Face Transformers, and PyTorch.
