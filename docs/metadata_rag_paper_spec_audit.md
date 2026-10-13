# Metadata-RAG Paper-to-Spec Audit

**Status:** Step 73 paper-to-spec audit
**Disposition:** `CONDITIONAL_GO` (ARCHITECTURAL_COMPARATOR, PAPER_DERIVED_REIMPLEMENTATION).

## Architecture Identity

- Paper: *Metadata-Driven Retrieval-Augmented Generation for Financial Question Answering*
- arXiv: `2510.24402`
- Year: 2025
- Venue: Preprint under major revision
- Registry verification: `PAPER_VERIFIED`

Identity is **resolved**. The only caveat is that the paper is a
preprint under major revision; its algorithmic claims are the design
reference but no paper-level effectiveness claim will be made against
this baseline until the paper is published.

## Why No Repository Is Cloned

The registry records `repository.official: false`, `url: null`, and
`license_spdx: NOT_APPLICABLE`. There is no foreign repository for
this baseline, and no foreign source exists to inspect or clone.

Under default copyright, no foreign code is needed, reused, or
executed. The implementation path is a project-owned paper-derived
reimplementation, structurally similar to the PathRAG comparator but
without any foreign checkout.

## Paper-Level Algorithm

The metadata-driven multistage retrieval pipeline:
- canonical documents and chunks
- project-owned metadata records
- query-derived metadata constraints or preferences
- metadata candidate selection
- semantic retrieval
- fused ranking
- canonical document expansion
- common evaluator
Final answer generation is out of scope. The pipeline ends at ranked
documents with full lineage.

## Metadata Schema and Value Semantics

The schema is not pinned by the paper. It must be designed
independently and frozen before any implementation:

- Value types: categorical, temporal, numeric, entity, structural,
  provenance.
- Field name normalization and value normalization are required.
- Missing-value, multi-value, and unknown-field policies are required.
- Typed comparison, range comparison, exact matching, and soft matching
  must be defined.
- Schema versioning is required so that later extensions do not
  invalidate frozen artifacts.

## Query Metadata Inference

Classification: **DETERMINISTIC_PROJECT_OWNED**.

The registry marks `offline_llm_required: true` for the paper's own
pipeline, but the retrieval-only project-owned reimplementation can
infer query metadata deterministically through rules or a frozen
parser. No hosted LLM is required for the retrieval boundary.

## Metadata and Semantic Retrieval Fusion

The paper defines a metadata-driven multistage retrieval policy but
does not pin the fusion coefficients or formula. These must be
designed independently and frozen under the same freeze discipline
used for the other baselines.

Required design decisions:

- Fusion policy (pre-filter, post-rerank, weighted score, RRF, etc.).
- Score ranges and normalization.
- Coefficients.
- Fallback when no metadata match or metadata missing.
- Deterministic tie-break.

## Retrieval-Only Boundary

Status: **DESIGN_REQUIRED**.

Because this is a project-owned implementation, the boundary is fully
definable. The design must expose, per ranked item:

- metadata-match evidence
- semantic score
- fused score
- canonical source IDs

before any generation stage.

## Canonical Source Traceability

Status: **REQUIRES_INDEPENDENT_DESIGN**.

Required mapping chain:
- metadata record
- canonical chunk ID
- canonical parent ID
- canonical document ID
Duplicate content and multiple documents per content must be
preserved. Score components must be exposed so downstream evaluation
can be reproduced.

## Dataset Compatibility and Leakage Prevention

The baseline must accept:

- provided metadata
- derived metadata (project-owned extraction)
- heterogeneous schemas
- metadata missingness

The following are **prohibited** as metadata sources:

- gold judgments
- reference answers
- evaluation labels
- generated answers
- post-retrieval annotations

## Resource Feasibility

| Stage | Status | Reason |
|---|---|---|
| Metadata ingestion | FEASIBLE_ON_LOCAL_HOST | text-only |
| Deterministic normalization | FEASIBLE_ON_LOCAL_HOST | pure-Python |
| Metadata index | FEASIBLE_ON_LOCAL_HOST | small synthetic fixture |
| Dense embedding | CONDITIONAL_ON_MODEL | CPU feasible for small pilot |
| Query metadata inference | FEASIBLE_ON_LOCAL_HOST | deterministic rules |
| Hybrid retrieval | FEASIBLE_ON_LOCAL_HOST | no service stack |
| Final generator | OUT_OF_SCOPE | retrieval-only boundary |
| **Overall** | **CONDITIONAL_ON_MODEL** | embedding model is the only dependency |

## Classification

- Architecture class: `ARCHITECTURAL_COMPARATOR`
- Implementation class: `PAPER_DERIVED_REIMPLEMENTATION`
- Foreign execution decision: `NOT_APPLICABLE`
- Project-owned implementation decision: `CONDITIONAL_GO`

## Terminal Disposition

**`CONDITIONAL_GO`.**

Paper identity is resolved and no foreign source is needed. The
reimplementation is feasible on the local host. The paper's fusion
formula and metadata schema are not fully pinned, so independent
design and freeze are required before readiness.

## Prohibited Claims

- Paper-to-spec audit is not an execution result.
- Project-owned implementation does not establish foreign-code parity.
- Synthetic validation is not paper-level effectiveness evidence.
- Metadata derived from judgments, reference answers, or evaluation
  labels is prohibited.
- READY_FOR_DATA requires typed artifacts and an end-to-end smoke.
- No foreign metadata_rag repository exists; no foreign source was
  inspected.

## Next Action

Write `baseline_audit/metadata_rag_clean_room_spec.yaml` freezing the
metadata schema, filtering semantics, retrieval fusion policy, and
canonical mapping chain. Then implement typed metadata artifacts, a
deterministic metadata index, a hybrid-ranking smoke, and a readiness
declaration.
