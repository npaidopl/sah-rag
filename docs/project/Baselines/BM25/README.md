# BM25 baseline (`bm25`)

Official BM25 from **Pyserini/Anserini** (Lucene backend), used as the lexical
reference baseline in SAH-RAG v2.

## Reference

- **Implementation:** https://github.com/castorini/pyserini
- **Paper:** Lin, Ma, Lin, Yang, Pradeep, Nogueira.
  *Pyserini: An Easy-to-Use Python Toolkit to Support Replicable IR Research
  with Sparse and Dense Representations.* arXiv:2102.10073 (2021).
- **Lucene BM25Similarity:** default parameters `k1=0.9`, `b=0.4`.

## License

**Apache-2.0** (Pyserini, Anserini, Lucene). Permissive; no legal obstacle for
inclusion as a research baseline.

## Status

`EXECUTABLE` (we have code, we can run it).

## Modes supported

| Mode | Supported | Notes |
|---|---|---|
| `native` | β… | BM25 as-is, official Lucene backend, no injected components used |
| `controlled` | β… (with exception) | BM25 accepts `embedder`/`llm` kwargs (required by frozen `BaseBaseline`) but **ignores** them, because BM25 is lexical. See below. |

### Structural exception in `controlled` mode

The frozen `BaseBaseline.__init__` requires accepting `embedder` and `llm`.
For BM25, both are **structurally irrelevant** β€” BM25 has no dense embedder and
no LLM generation. The adapter accepts them and does not use them. This is not a
contract violation; it reflects the baseline's architecture and must be declared
when reporting controlled-mode results.

## Dependencies

- `pyserini` (pip, see `requirements.txt`)
- **Java 21 runtime** (JDK required for Anserini/Lucene). Not a pip dependency.
- Lucene 9.x jars (bundled by Anserini β€” no separate install).

## Index strategy

`build_index(documents)` serializes the corpus to `work_dir/corpus_jsonl/corpus.jsonl`
(`{"id": ..., "contents": ...}` per line), then invokes the **official Pyserini
indexer CLI**:
python -m pyserini.index.lucene ^
--collection JsonCollection ^
--input <work_dir>/corpus_jsonl ^
--index <work_dir>/lucene_index ^
--generator DefaultLuceneDocumentGenerator ^
--threads 1 ^
--storePositions --storeDocvectors --storeRaw

The resulting Lucene index persists under `work_dir/lucene_index/` and is
loaded with `LuceneSearcher` at query time.

**Idempotent:** calling `build_index` again rebuilds from scratch (directories
are cleared first).

## Retrieval output

- `doc_id` = Lucene document id = `CanonicalDocument.doc_id`.
- `chunk_id` = same as `doc_id` (BM25 has no chunking).
- `score` = Lucene BM25 score (higher = better).
- `rank` = contiguous 1..N.

## Known limitations

1. **`k1` / `b` overrides are not applied.** The Pyserini CLI uses Lucene
   defaults (`k1=0.9`, `b=0.4`). Custom `Similarity` requires a Pyserini fork
   (out of scope). Overrides in `extra:` are recorded but warn at init.
2. **`embedder` / `llm` are ignored in controlled mode.** Documented above.
3. **Document-level only.** No sub-document chunking; `chunk_id == doc_id`.
4. **JVM required.** The test skips if `java` is not on `PATH`.

## Reproduction check (native mode)

After building the index, the following must match the official Pyserini
regression numbers for the chosen dataset (once a dataset is selected):

- `pyserini.search.lucene.LuceneSearcher` on the same index
- Default Lucene BM25Similarity
- Same tokenization (Lucene StandardAnalyzer, JSONL collection)

Discrepancies imply version drift in `pyserini`, Anserini or Lucene and must be
resolved by pinning versions.

---

## Windows / pytest faulthandler (known issue, resolved)

On Windows, `pytest`'s default `faulthandler` plugin emits false-positive
`Windows fatal exception: access violation` traces when `jnius` (the
Python-JVM bridge used by Pyserini) initializes the JVM. The JVM uses SIGSEGV
internally for safepoint polling, which `faulthandler` interprets as a crash.

**Resolution:** `baselines/bm25/tests/conftest.py` unregisters the
`faulthandler` plugin at `pytest_configure` time. This removes the noise
without hiding real failures (non-zero exit still propagates).

If you need to run the test in a context where `conftest.py` is not loaded
(e.g., from a different `--rootdir`), pass `-p no:faulthandler` explicitly.

### Timing note

Cold-run wall time is ~16-20s on this machine:
- JVM startup: ~10s
- Pyserini indexing (5 docs): ~3s
- 3 queries + teardown: ~3s
- Python + pytest startup: ~3s

Subsequent runs are typically faster (~12-15s) because the JVM and Lucene
class cache are warm. This is acceptable for a small test corpus; real
benchmark runs will amortize the JVM cost over the whole dataset.


### Correction (supersedes the note above)

The initial `conftest.py` only unregistered the pytest `faulthandler` plugin,
which was too late (pytest enables faulthandler very early). The current
`conftest.py` monkeypatches `faulthandler.enable` to a no-op at *module import*
time of `conftest.py`, before any pytest hook runs, and calls
`faulthandler.disable()` eagerly. This reliably silences the false-positive
"Windows fatal exception: access violation" traces.

This still does NOT hide real failures: process-level crashes still propagate
as non-zero pytest exit codes, and assertion failures are reported normally.


---

## Arm registry

See `ARM_REGISTRY.md` for the three arm IDs this baseline occupies
(`BM25_NATIVE`, `BM25_CONTROLLED`, `BM25_REFERENCE`), why `NATIVE` and
`CONTROLLED` are functionally identical for BM25, and the graduation checklist.
