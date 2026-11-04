# Security Incident Report - 2026-11-04

## Summary

A Hugging Face API token (prefix hf_Wxwbo..., full token redacted) was exposed
in a prior LLM chat session log during PathRAG wrapper debugging.
The token had read-only access. This doc records the incident and remediation.

Severity: Low (read-only, no evidence of misuse, not in git).

## Timeline

- 2026-10-31: Token created for huggingface-cli downloads.
- 2026-11-03: Token exposed in LLM chat transcript.
- 2026-11-04: Token identified during PathRAG review. Cell 1 scan confirmed
  token NOT in repo text files (pattern hf_Wxwbo, excluding .venv, .git, __pycache__).
- 2026-11-04: Token revoked at https://huggingface.co/settings/tokens (manual).
- 2026-11-04: .env / .env.example scanned; leaked token replaced if present.

## Impact

- Access scope: read-only to gated models. No write, no admin.
- Exposure: chat transcript only; not in git, not in committed files.
- Misuse evidence: none.

## Remediation

1. Token revoked at HF dashboard.
2. .env / .env.example scanned for hf_Wxwbo prefix.
3. Repo-wide text search - zero matches.
4. Diagnostic pattern added to future sessions.

## Prevention

1. Never paste tokens in LLM prompts. Use env vars.
2. Use .env (gitignored) for secrets.
3. New tokens: read-only scope, stored in .env only.
4. Pre-commit hook (future): block hf_ patterns.

## Verification

Run in future sessions:

    Get-ChildItem -Recurse -Include '*.py','*.md','*.yaml' | Select-String -Pattern 'hf_Wxwbo'
    git log -p -S 'hf_Wxwbo' --all

Both should return zero matches.

## Sign-off

- Reported: SAH-RAG orchestrator (2026-11-04)
- Revoked: project owner (manual)
- Verified: Cell 1 diagnostic (zero matches)
- Next review: before first GPU execution
