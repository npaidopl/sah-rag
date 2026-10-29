
---

## Candidate 2 - Upper Context for Uncued Visual Units

**What HiKEY does (Section 3.1):** For table/figure units without captions, traverses the DHP tree to find the logically preceding text node and stores it as Upper Context. Helps reduce visual ambiguity.

**Why it may matter:** A table or figure without a caption is often unintelligible in isolation. Upper context is cheap and principled.

**SAH-RAG ablation variant:** If SAH-RAG has its own handling of uncued units (or ignores them), add a variant that uses Upper Context. Compare.

**Expected outcome:** If SAH-RAG-default wins, SAH-RAG's mechanism is superior. If Upper Context wins, it is a candidate addition.

**Cost:** ~100 lines (tree traversal + serialization). Requires SAH-RAG to support multimodal units.

---

## Candidate 3 - Ancestry-Aware Evidence Assembly

**What HiKEY does (Section 3.3, Appendix E):** When selecting an evidence unit, always serialize its governing headers (the ancestor chain of section titles) alongside the unit. This prevents scope confusion in the reader.

**Why it may matter:** Evidence without scope is ambiguous. "Temperature reaches 3000C" means something different under "Solar Corona" vs "Furnace Test".

**SAH-RAG ablation variant:** Add governing headers to SAH-RAG evidence serialization if it does not already. Compare.

**Expected outcome:** If SAH-RAG-default already does this (or if the reader does not benefit), the ablation is confirmation. If it wins, it is a candidate addition.

**Cost:** ~50 lines.

---

## Recommendation

Start with **Candidate 1 (MaxSim)**. It is:
- Simplest to implement
- Most directly relevant to retrieval metrics
- Lowest risk of changing SAH-RAG's core identity

Candidates 2 and 3 are secondary. Include them only if Candidate 1 shows meaningful signal, or if SAH-RAG's current design already implements them (in which case, the comparison becomes a confirmation rather than an addition).

---

## What NOT to do

- Do not rebrand SAH-RAG as HiKEY-inspired. The ablation is about component-level comparison, not architectural fusion.
- Do not use these ablations to claim SAH-RAG beats HiKEY. HiKEY comparisons must use paper-reported numbers (see `baselines/hikey/reported_metrics.yaml`).
- Do not include HiKEY's DHP or multimodal encoders in SAH-RAG as part of this ablation. Those are whole-architecture additions, not component-level ablations.

---

## References

- HiKEY paper: arXiv:2605.29606v1 (Shin et al., ACL 2026)
- SAH-RAG main architecture: `sah_rag/`
- Reported metrics: `baselines/hikey/reported_metrics.yaml`