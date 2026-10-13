# Documentation

This directory contains the written documentation of the SAH-RAG project.

## Contents

### Reports

Located under `data/`. See [`data/README.md`](data/README.md) for details.

- `REPORT_NIKOLAOS_PAIDOPOULOS.pdf` — full Diploma Thesis
- `metadata_rag_full_track_report.tex` — Metadata-RAG comparator report
- `three_way_comparison.tex` — three-way comparison report

### Methodology

Frozen methodology artifacts are stored in the sibling
[`baseline_audit/`](../baseline_audit/) directory. Key contracts include:

- `common_retrieval_evaluation_contract.yaml`
- `cross_baseline_execution_contract.yaml`
- `fair_benchmark_*.yaml`
- `statistical_analysis_plan.yaml`
- `component_substitution_policy.yaml`

These YAML contracts are **frozen**: they are read by validators and are not
modified after the corresponding experiment has been committed.

### Architecture

The architecture is documented in:

- Top-level [`README.md`](../README.md) — overview and quick start
- Source docstrings under [`src/rag_system/`](../src/rag_system/)
- Thesis Chapter 3 (design) and Chapter 4 (implementation), in `data/`

### Reproducibility

Every run produces a `manifest.json` with SHA-256 checksums of all inputs and
outputs. To verify a run:

    python scripts/validate_*.py --manifest path/to/manifest.json

See the individual scripts under [`scripts/`](../scripts/) for details.

## Navigating the codebase

| I want to... | Start here |
|---|---|
| Understand the architecture | top-level `README.md` |
| Run the full pipeline | `scripts/run_my_rag_*.py` |
| Reproduce the comparison | `scripts/compare_*.py` |
| Inspect frozen contracts | `baseline_audit/*.yaml` |
| Read the thesis | `docs/data/REPORT_NIKOLAOS_PAIDOPOULOS.pdf` |
| Verify a specific artifact | the corresponding `manifest.json` |
