# Stage 2 Report β€” MS MARCO passage v1 BM25 Reproduction (Credibility Check)

**Date:** 2026-10-22 22:40:21
**Environment:** Java 21 (jdk-21.0.12.101-hotspot), pyserini==2.4.0, pytrec_eval_terrier 0.5.11, Python 3.12
**Purpose:** Credibility check β€” validate that the Pyserini/Lucene/BM25 stack
scales from BEIR (5K docs) to MS MARCO (8.8M docs) and reproduces the
canonical MRR@10. **Not** the (Ξ±) native E2E deliverable.

## Cache location

All Pyserini data on D: (PYSERINI_CACHE=D:\rag_work\pyserini_cache).

## Search command
python -m pyserini.search.lucene ^
--index msmarco-passage ^
--topics msmarco-passage-dev-subset ^
--output baselines/bm25\runs\msmarco_passage_v1.bm25.trec ^
--output-format trec --bm25 --hits 1000 --threads 4 --batch 36 --remove-query

## Evaluation (MRR@10)
python baselines/bm25/scripts/eval_ndcg.py ^
baselines/bm25/runs/msmarco_passage_v1.bm25.trec ^
msmarco-passage-dev-subset recip_rank 10

## Result

| Metric | Value |
|---|---|
| MRR@10 (ours) | 0.187486 |
| MRR@10 (published, Pyserini regression) | 0.187 |
| Delta | 0.00049 |
| Tolerance | 0.005 |
| **Verdict** | **PASS** |

## Artifacts

- Run file: baselines/bm25/runs/msmarco_passage_v1.bm25.trec (6974596 lines, 274.2 MB)
- Eval script: baselines/bm25/scripts/eval_ndcg.py
- Search elapsed: 57.7 min
- Eval elapsed: 9.8 s

## Scope note

This is a **credibility check** of the environment, not the (Ξ±) native E2E
deliverable. The (Ξ±) deliverable requires the BM25 adapter to run on the
**SAH-RAG dataset** (not yet selected). MS MARCO confirms that Java 21 +
Pyserini 2.4.0 + Lucene + BM25 reproduce the canonical baseline at scale.

## Interpretation

Pyserini/Lucene/BM25 stack scales correctly to MS MARCO (8.8M docs).
MRR@10 = 0.187 reproduced within tolerance. Environment fully validated.