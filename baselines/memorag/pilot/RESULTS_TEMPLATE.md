# MemoRAG pilot — results

> Copy this template to `pilot/artifacts/<run_id>/RESULTS.md` after each
> run and fill in every section. Do not delete sections; write "n/a"
> with a reason if a section does not apply.

## Run metadata

| Field | Value |
|---|---|
| pilot_id | `<native_pilot_v1 \| controlled_pilot_v1>` |
| run_id | `<timestamp>` |
| operator | `<name>` |
| host | `<hostname>` |
| gpu | `<model>, <vram_gb> GB` |
| date_utc | `<YYYY-MM-DD>` |
| manifest | `pilot/<manifest>.yaml` |
| memorag_commit | `<hash>` |
| package_version | `<version>` |
| exit_code | `<0 \| 1 \| 2 \| 3 \| 4>` |

## Environment (from env.txt)

- python: `<version>`
- platform: `<platform>`
- torch: `<version>`
- cuda_available: `<true | false>`
- cuda_device_names: `<...>`
- cuda_memory_gb: `<...>`

## Native run

### Artifact checklist

| File | Present | Notes |
|---|---|---|
| `env.txt` | ☐ | |
| `probe.json` | ☐ | |
| `methods.json` | ☐ | |
| `load.log` | ☐ | |
| `retrieve_probe.json` | ☐ | |
| `generate_probe.json` | ☐ | only if generator.enabled |
| `run.json` | ☐ | |

### Observations

- memorag import: `<ok | fail: message>`
- memorag package file: `<path>`
- memorag version: `<version>`
- memory model id: `<id>@<revision>`
- retriever model id: `<id>@<revision>`
- memory model load time: `<seconds>`
- retriever load time: `<seconds>`
- corpus size: `<n>` docs
- retrieval elapsed: `<seconds>`
- retrieval returned: `<n>` items
- generation enabled: `<true | false>`
- generation elapsed: `<seconds>` (if enabled)
- generation output preview: `<first 200 chars>` (if enabled)

### Verdict

`<PASS | FAIL>` — `<one-line summary>`

### Issues

- `<list any warnings from load.log, probe.json, or methods.json>`

## Controlled run

### Artifact checklist

| File | Present | Notes |
|---|---|---|
| `env.txt` | ☐ | |
| `probe.json` | ☐ | |
| `load.log` | ☐ | |
| `retrieve_probe.json` | ☐ | |
| `run.json` | ☐ | must show `generation_executed: false` |
| `generate_probe.json` | ☐ | **must be absent** |

### Observations

- memorag import: `<ok | fail: message>`
- memory model id: `<id>@<revision>` (native, unchanged)
- injected embedder id: `<id>@<revision>` (must equal BGE-M3 revision)
- injected llm id: `<id>@<revision>` (not used for generation)
- generation executed: `<false>` (contract requirement)
- memory model load time: `<seconds>`
- corpus size: `<n>` docs
- retrieval elapsed: `<seconds>`
- retrieval returned: `<n>` items

### Verdict

`<PASS | FAIL>` — `<one-line summary>`

### Issues

- `<list any warnings>`

## Cross-run comparison

| Aspect | Native | Controlled |
|---|---|---|
| memory model id | `<id>@<rev>` | `<id>@<rev>` |
| retriever / embedder id | `<id>@<rev>` | `<id>@<rev>` |
| generation executed | `<true \| false>` | `false` |
| retrieval returned | `<n>` | `<n>` |
| elapsed (total) | `<sec>` | `<sec>` |

Interpretation: `<if memory id and embedder id match across runs, the
controlled run is a fair controlled-adaptation reference; if they
differ, the difference must be documented and justified here>`

## Proof of nativeness (five-point evidence)

This section is what an examiner will read to verify the native claim.
Each point must be supported by a concrete artifact reference, not by
prose.

### 1. Exact source

- memorag commit: `<hash>` — evidence: `probe.json` + manifest
- package file: `<path>` — evidence: `probe.json`
- python executable: `<path>` — evidence: `env.txt`
- working directory: `<path>` — evidence: `env.txt`

### 2. Real checkpoint loaded

- memory model id: `<id>@<revision>` — evidence: `load.log`
- load time: `<seconds>` — evidence: `load.log`
- sanity: a 7B checkpoint cannot load in < 1s; measured time is
  consistent with a real load: `<yes | no>`

### 3. Clue stage executed

- retrieval in MemoRAG is clue-conditioned; empty output implies
  failure of the pipeline, not absence of the clue stage
- `retrieve_probe.json` returned: `<n>` items
- `retrieve_probe.json` elapsed: `<seconds>`
- evidence that clues were generated: `<quote log line, or describe
  observation from retrieve_probe.json>`

### 4. Generation on/off explicit

- generator.enabled in manifest: `<true | false>`
- `generate_probe.json` present: `<yes | no>`
- consistency: `<for native e2e, must be true/yes; for controlled,
  must be false/no>`

### 5. No substitutions

- native mode: no injected embedder, no injected LLM in the manifest
  — evidence: `native_manifest.yaml`
- controlled mode: only the embedder is injected; memory model
  remains the native checkpoint — evidence: `controlled_manifest.yaml`
  and `load.log` memory model id line

**Overall native-ness verdict**: `<PASS | FAIL>`
**Examiner-facing one-liner**: `<one sentence tying artifacts to the
author-supported native claim>`

## Anti-drift check

The following values must match between this run and the previous
accepted run. Any change requires a new pilot_id, per
`PILOT_README.md`.

| Field | Current | Previous | Match |
|---|---|---|---|
| memorag commit | `<...>` | `<...>` | ☐ |
| memory model id | `<...>` | `<...>` | ☐ |
| memory model revision | `<...>` | `<...>` | ☐ |
| retriever id | `<...>` | `<...>` | ☐ |
| retriever revision | `<...>` | `<...>` | ☐ |
| generator id (if enabled) | `<...>` | `<...>` | ☐ |
| generator revision (if enabled) | `<...>` | `<...>` | ☐ |
| context.max_tokens | `<...>` | `<...>` | ☐ |
| context.clue_count | `<...>` | `<...>` | ☐ |

If any row shows a mismatch, do not treat this run as the same
pilot. Assign a new pilot_id and record the change in the manifest.

## Next steps

- [ ] If native verdict is PASS: this baseline is authorized to
      receive real datasets once they are frozen.
- [ ] If any artifact is missing or the verdict is FAIL: fix the
      smoke test or environment; do not proceed to datasets.
- [ ] Attach this RESULTS.md to the pilot artifacts directory so it
      travels with the run.
- [ ] Update `baselines/memorag/README.md` "Known limitations" if the
      pilot reveals a limitation worth recording.

## Sign-off

- Operator: `<name>`
- Date (UTC): `<YYYY-MM-DD>`
- Signature/initials: `<initials>`
- Reviewer (if applicable): `<name>`