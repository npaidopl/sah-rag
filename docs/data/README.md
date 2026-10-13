# Reports & Manuscripts

This directory contains the written artifacts of the SAH-RAG project.

## Files

| File | Description |
|------|-------------|
| REPORT_NIKOLAOS_PAIDOPOULOS.pdf | Full diploma thesis (Greek + English abstract). |
| metadata_rag_full_track_report.tex | LaTeX source of the Metadata-RAG full-track comparison report (Ch. 6 companion). |
| three_way_comparison.tex | LaTeX source of the three-way comparison report (SAH-RAG vs LongRAG vs Metadata-RAG). |

## Compiling the LaTeX sources

The .tex files assume a standard LaTeX toolchain with latexmk:

    cd docs/data
    latexmk -pdf metadata_rag_full_track_report.tex
    latexmk -pdf three_way_comparison.tex

Compiled PDFs are ignored by .gitignore (they are regenerable).

## Related external resources

- Canonical dataset suite: https://github.com/npaidopl/canonical-rag-dataset-suite
- Project repository: see top-level README.md
