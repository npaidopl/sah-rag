# LongRAG vs My_RAG — Comparative Retrieval Report

## Protocol

- **Scope**: 4 datasets (clerk, cuad, greek_bar_bench, lara)
- **Embedder**: BAAI/bge-m3 @ 5617a9f6... (identical)
- **Precision**: FP16 (identical)
- **Evaluator**: document-level, cutoffs 1/5/10/20/50 (identical)
- **Ground truth**: canonical relevant_ids (identical)
- **Retrieval-only**: yes; generation disabled

## Primary metric

**macro_document_Recall@10**: LongRAG = `0.3144`, My_RAG = `0.4049`, delta = `+0.0905` (`+28.79%`)

## Macro-level comparison

| Metric | LongRAG | My_RAG | Delta | Winner |
|---|---:|---:|---:|---|
| hit@1 | 0.1957 | 0.2553 | +0.0596 | my_rag |
| hit@5 | 0.2606 | 0.3846 | +0.1241 | my_rag |
| hit@10 | 0.3183 | 0.4520 | +0.1338 | my_rag |
| hit@20 | 0.3881 | 0.5416 | +0.1535 | my_rag |
| hit@50 | 0.4929 | 0.6658 | +0.1729 | my_rag |
| recall@1 | 0.1905 | 0.2180 | +0.0275 | my_rag |
| recall@5 | 0.2569 | 0.3394 | +0.0825 | my_rag |
| recall@10 | 0.3144 | 0.4049 | +0.0905 | my_rag |
| recall@20 | 0.3833 | 0.4938 | +0.1105 | my_rag |
| recall@50 | 0.4870 | 0.6210 | +0.1340 | my_rag |
| precision@10 | 0.0362 | 0.0557 | +0.0195 | my_rag |
| mrr@1 | 0.1957 | 0.2553 | +0.0596 | my_rag |
| mrr@5 | 0.2193 | 0.3032 | +0.0840 | my_rag |
| mrr@10 | 0.2268 | 0.3123 | +0.0855 | my_rag |
| mrr@20 | 0.2316 | 0.3186 | +0.0870 | my_rag |
| mrr@50 | 0.2349 | 0.3225 | +0.0876 | my_rag |
| ndcg@10 | 0.2476 | 0.3214 | +0.0737 | my_rag |
| ndcg@50 | 0.2880 | 0.3733 | +0.0853 | my_rag |

## Per-dataset comparison

| Dataset | LongRAG R@10 | My_RAG R@10 | Delta | Winner | Evaluated |
|---|---:|---:|---:|---|---:|
| clerk | 0.0786 | 0.3546 | +0.2760 | my_rag | 1000 |
| cuad | 0.0324 | 0.0431 | +0.0107 | my_rag | 6702 |
| greek_bar_bench | 0.2606 | 0.2852 | +0.0246 | my_rag | 284 |
| lara | 0.8861 | 0.9368 | +0.0507 | my_rag | 2326 |

## Interpretation

LongRAG's K-consecutive-documents grouping policy produces large units that mix evidence from multiple documents. When the source corpus does not exhibit Wikipedia-style hyperlink adjacency and documents are short (CLERC), a single relevant passage is diluted by ten unrelated passages after expansion to documents. My_RAG's hierarchical child chunking with parent aggregation preserves fine-grained evidence and yields much stronger document-level ranking on such corpora.

On corpora whose documents are naturally large (LaRA), both architectures perform well and the gap narrows. This is consistent with the architectural observation that grouping is more valuable when the source units are themselves the retrievable evidence.

## Reproducibility

- LongRAG closure manifest: `baseline_audit/longrag_full_track_closure_manifest.yaml`
- My_RAG closure manifest: `baseline_audit/my_rag_m0_closure_manifest.yaml`
- Comparison manifest: `baseline_audit/longrag_vs_my_rag_comparison.yaml`