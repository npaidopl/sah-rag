# External Canonical Dataset Suite Registration

**Step:** 92
**Status:** COMPLETE
**Scope:** Read-only registration and evaluation-semantics freeze.

## Purpose

Register the seven real long-context datasets that live in an
external project root, freeze the collection identity, and audit
corpus scope and ground-truth projection. No dataset is downloaded,
no adapter is executed, no parser/chunker/tokenizer/model/embedding
runs.

## External layout
${SAH_RAG_PROJECT_ROOT}/
β”β”€β”€ processed/canonical/ β† 7 canonical datasets
β”β”€β”€ datasets/ β† raw sources
β””β”€β”€ pipeline/ β† adapters

External repository:

- URL: `https://github.com/npaidopl/canonical-rag-dataset-suite`
- HEAD commit: `8c3cdc6a2030235de2955b4f832c7a3937d6483a`
- Collection fingerprint:
  `5f577978b6ab7a4ee3fef29b68db29f95b37261f103bb6cfd5fd7094c70a2aa4`

## Registered datasets

| Dataset | Documents | Examples | Scope | Primary unit |
|---|---:|---:|---|---|
| ruler | 4,000 | 4,000 | QUERY_SPECIFIC_CORPUS | NEEDLE_OR_EVIDENCE_ITEM |
| loong | 695 | 695 | QUERY_DOCUMENT_BUNDLE | CANONICAL_DOCUMENT |
| long_bench_v2 | 503 | 503 | QUERY_SPECIFIC_CORPUS | CANONICAL_DOCUMENT |
| lara | 131 | 2,326 | SHARED_DOMAIN_CORPUS | CANONICAL_DOCUMENT |
| greek_bar_bench | 284 | 284 | QUERY_SPECIFIC_CORPUS | SOURCE_SPAN |
| cuad | 510 | 20,910 | SHARED_DOMAIN_CORPUS | SOURCE_SPAN |
| clerk | 2,756 | 1,000 | GLOBAL_DATASET_CORPUS | PASSAGE |
| **Total** | **8,879** | **29,718** | | |

## Registration verification

- Repository commit resolved from `git rev-parse HEAD`
- All 28 canonical files (4 per dataset) SHA-256 computed
- File hashes match the published `canonical_manifest.json`
- All 7 dataset manifests valid (`raw_source_modified: false`)
- All 7 validation reports PASS (zero errors)
- Document IDs unique per dataset
- Query IDs unique per dataset
- `query.doc_ids` references resolve to canonical documents
- `ground_truth.relevant_ids` references resolve
- Collection fingerprint matches expected value
- External canonical suite unchanged after registration

## Corpus-scope classification

Four scope classes are frozen:

- **QUERY_SPECIFIC_CORPUS**: ruler, long_bench_v2, greek_bar_bench
- **QUERY_DOCUMENT_BUNDLE**: loong
- **SHARED_DOMAIN_CORPUS**: lara, cuad
- **GLOBAL_DATASET_CORPUS**: clerk

RULER carries a HIGH trivial-document-selection risk because each
example contains its own per-example context. Document-level
retrieval alone would be trivially perfect. The primary evaluation
unit for RULER is therefore NEEDLE_OR_EVIDENCE_ITEM.

## Ground-truth projection classification

| Primary unit | Datasets |
|---|---|
| CANONICAL_DOCUMENT | loong, long_bench_v2, lara |
| SOURCE_SPAN | greek_bar_bench, cuad |
| NEEDLE_OR_EVIDENCE_ITEM | ruler |
| PASSAGE | clerk |

Global rules:

- Ground truth is available **only** to the evaluator.
- Ground truth is **not** available to retrieval, indexing, or querying.
- Answer fields are prohibited in the retrieval pipeline.
- Duplicate document identities remain distinct.

CUAD explicitly supports zero-relevance queries (14,208 unanswerable
clause QAs). The evaluation contract at Step 93 must accommodate
`relevant_item_count == 0` without rejecting the case.

## Safety boundary

No adapter, parser, chunker, tokenizer, model, embedding, Qdrant,
retrieval, evaluation, or generation was executed during Step 92.

## Next gate

**Step 93: Fair Benchmark Protocol and Architecture Fidelity Matrix.**

This is the methodological checkpoint that freezes the native vs
controlled component policy, the LLM policy, the storage
substitution policy, the precision policy, and the corpus isolation
policy before any architecture implementation begins.