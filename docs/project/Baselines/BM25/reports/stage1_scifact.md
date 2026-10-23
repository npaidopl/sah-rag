# Stage 1 Report β€” BEIR SciFact BM25 Reproduction Check

**Date:** 2026-10-22 21:30:10
**Environment:** Java 21 (jdk-21.0.12.101-hotspot), pyserini==2.4.0, pytrec_eval_terrier 0.5.11, Python 3.12

## Correction note

The initial run compared against **0.665**, which is the value reported in the
**BEIR paper** (arXiv:2104.08663, Table 2). That value came from a different
implementation and does not correspond to the eir-v1.0.0-scifact.flat
pre-built index used here.

The correct reference for the **flat** index is the official Anserini
regression value: **0.6789** (see nserini/docs/regressions/regressions-beir-v1.0.0-scifact-flat.md).
Our result is **0.678909**, delta **9E-06** β€” effectively identical.

## Search command (official Pyserini CLI)
python -m pyserini.search.lucene ^
--index beir-v1.0.0-scifact.flat ^
--topics beir-v1.0.0-scifact-test ^
--output baselines/bm25\runs\scifact.bm25.trec ^
--output-format trec --bm25 --hits 1000 --threads 1 --batch 36 --remove-query
## Evaluation command
python baselines/bm25/scripts/eval_ndcg.py ^
baselines/bm25/runs/scifact.bm25.trec ^
beir-v1.0.0-scifact-test ndcg_cut.10



## Result

| Metric | Value |
|---|---|
| nDCG@10 (ours) | 0.678909 |
| nDCG@10 (official flat regression) | 0.6789 |
| Delta vs flat | 9E-06 |
| Delta vs BEIR paper (different setup) | 0.0139 |
| Tolerance | 0.005 |
| **Verdict** | **PASS** |

## Artifacts

- Run file: baselines/bm25/runs/scifact.bm25.trec
- Eval script: baselines/bm25/scripts/eval_ndcg.py

## Interpretation

Environment verified. Our BM25 run on the official lat index reproduces the
official Anserini regression value (0.6789) to within 1e-5. The 0.665 value is
a BEIR-paper figure from a different implementation and is not the correct
reference for this index.

**Stage 1 complete.** Ready for Stage 2 (MS MARCO passage v1).