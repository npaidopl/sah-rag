# Written Reports

LaTeX sources of all written artifacts of the SAH-RAG project.

## Primary reports

| File | Topic |
|---|---|
| `sah_rag_implementation.tex` | SAH-RAG architecture and implementation |
| `sah_rag_optimization.tex` | SAH-RAG parameter study and frozen operating profile |
| `datasets.tex` | Canonical dataset suite description |
| `pipelines_status.tex` | Summary of pipelines and stages |

## Comparison reports

| File | Topic |
|---|---|
| `longrag_full_track_report.tex` | LongRAG controlled adaptation — full track |
| `longrag_vs_sah_rag.tex` | LongRAG vs SAH-RAG comparison |
| `metadata_rag_full_track_report.tex` | Metadata-RAG paper-derived comparator — full track |
| `three_way_comparison.tex` | SAH-RAG vs LongRAG vs Metadata-RAG |

## Archived reports (`archived/`)

Status reports on architectures that were evaluated for inclusion but
ultimately not executed under the shared protocol. They document the
reasons for exclusion.

| File | Architecture |
|---|---|
| `disretrieval_status.tex` | DISRetrieval |
| `lightrag_report.tex` | LightRAG |
| `pathrag_status.tex` | PathRAG |
| `raptor_hipporag_lightrag_status.tex` | RAPTOR, HippoRAG, LightRAG |

See also [`remaining_baselines_sequencing.md`](../remaining_baselines_sequencing.md)
for the methodological note on exclusions.

## Compiling

The `.tex` sources assume a standard LaTeX toolchain with `latexmk`:

    cd docs/reports
    latexmk -pdf sah_rag_implementation.tex
    latexmk -pdf three_way_comparison.tex

Compiled PDFs are ignored by `.gitignore` (they are regenerable).

## Naming note

Earlier drafts referred to the SAH-RAG architecture as "My_RAG" (the internal
working name). Renamed files now use "SAH-RAG" throughout. Frozen
experimental contracts under `baseline_audit/` retain the historical name
`my_rag_*` to preserve SHA-256 reproducibility.
