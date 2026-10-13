# Remaining Baselines — Sequencing Decision

**Status:** preliminary portfolio preflight (Step 69)
**Scope:** DisRetrieval, MemoRAG, KAG, Metadata-RAG, HiKEY
**Checkouts present:** none

---

## Decision

Wave-1 full publishable execution is **not** started yet. All ten
architecture slots must first reach a documented terminal planning
disposition. Terminal dispositions may include `READY_FOR_DATA` for
some slots and `DEFERRED_*` or `BLOCKED_*` for others. A slot that
remains `NOT_STARTED`, `UNKNOWN` or `PENDING_UNBOUNDED` blocks the
massive run.

This decision replaces the earlier assumption that all ten baselines
must reach `READY_FOR_DATA`. Contract stability matters more than
uniform execution coverage.

## Why freeze the contracts first

If we run Wave-1 baselines now and only later discover that HiKEY
requires multimodal input fields, the cross-baseline execution
contract must be bumped from `0.1.0` to `0.2.0`, and every Wave-1
result must be regenerated on the new contract to keep comparisons
fair. Freezing after all ten dispositions is cheaper than re-running.

## Preliminary dispositions (Step 69)

| Architecture | Preliminary status | Reason |
|---|---|---|
| DisRetrieval | `PREFLIGHT_CONDITIONAL_GO` | License is `UNKNOWN`; architecture appears feasible |
| MemoRAG | `PREFLIGHT_CONDITIONAL_GO` | Hosted-LLM dependency expected; Apache-2.0 |
| KAG | `PREFLIGHT_CONDITIONAL_GO` | Hosted LLM + heavy graph; component audit required |
| Metadata-RAG | `PREFLIGHT_GO` | Project-owned reimplementation; no repo |
| HiKEY | `PREFLIGHT_DEFERRED_INPUT_CONTRACT` | Multimodal; no repo released; contract bump required |

These are preliminary. They are derived from the registry and the
paper audit trail, not from source inspection. Each will be replaced
by a terminal disposition after its exact-source audit.

## Sequencing

1. **Step 70 — DisRetrieval** exact-source audit and disposition.
2. **Step 71 — MemoRAG** exact-source audit and disposition.
3. **Step 72 — KAG** exact-source audit and disposition.
4. **Step 73 — Metadata-RAG** project-owned implementation audit.
5. **Step 74 — HiKEY** deferred; audit only if a public repository is
   released and the multimodal input contract is designed.
6. **Step 75 — Registry terminal disposition freeze** for all ten
   slots.

Wave 2 is completed before Wave 3 because it is the next registered
group and it will surface hosted-LLM, long-context and graph
construction constraints earlier, which reduces the risk of a
contract bump after Wave-3 audit.

## What is allowed before the freeze

The following infrastructure work is permitted on the Wave-1
architectures before the terminal-disposition freeze:

- synthetic orchestrator dry-run smoke
- dataset envelope validation
- execution-manifest lifecycle tests
- resource estimator tests
- failure / resume simulation
- artifact transfer simulation

The following are **not** permitted before the freeze:

- full dataset indexing
- production embeddings
- hosted LLM extraction
- large graph construction
- large-scale retrieval runs
- final comparison tables
- reported benchmark metrics

If a tiny non-reportable infrastructure cell is executed, it must
carry `execution_role: INFRASTRUCTURE_VALIDATION_ONLY`,
`publishable_result: false`, and `included_in_final_comparison: false`.

## Conformance

The validator `scripts/validate_remaining_baselines_preflight.py`
enforces:

- exactly five entries, matching the five remaining baselines
- every entry has a `preliminary_status` from the allowed vocabulary
- no entry uses a forbidden terminal status
- every `PREFLIGHT_CONDITIONAL_GO` and `PREFLIGHT_DEFERRED_*` entry
  declares at least one `blockers` item
- every entry has a `next_action`
- the `sequencing_decision` block matches the reviewer decision
- the `planned_sequence` covers steps 69 through 75
