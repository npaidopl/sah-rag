# Prompt templates

These are the prompt templates used by `native_e2e.runner`. Each file is a
plain-text template with two placeholders:

    {context}   — concatenated retrieved long units
    {question}  — the user question

## Native-fidelity caveat

The templates here are **transcriptions from the LongRAG paper
(arXiv:2406.15319) Appendix A.1**, lightly normalised for a generic
chat-completions interface. **They have NOT been verified byte-for-byte
against the official repository.**

Before claiming `PAPER_EXACT_NATIVE` or `AUTHOR_SUPPORTED_NATIVE`, the
supervisor MUST:

1. Open `external/baselines/longrag/utils/gpt_inference.py` (and the
   Gemini variant) at the pinned commit.
2. Replace each template file with the exact prompt string.
3. Record the SHA-256 of each file in `baseline_audit/longrag_native_reader_config.yaml`.

The runner records the SHA-256 of every prompt file it loads into the run
manifest, so any later edit is visible in provenance.