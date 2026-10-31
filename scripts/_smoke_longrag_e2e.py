"""Smoke test: tiny dataset through native E2E and controlled pipelines.

Offline. Uses mock embedder + mock reader only to exercise the pipeline;
NOT a native-fidelity claim. Production runs use BGE-large + GPT-4o/Gemini.
"""

import json
import sys
import importlib.util
from pathlib import Path

from baselines.base import BaselineConfig, RunMode
from baselines.longrag_v2.adapter import LongRAGBaseline
from baselines.longrag_v2.native_e2e import E2ERunner, load_prompt
from baselines.longrag_v2.native_e2e.cli import main as cli_main
from sah_rag.canonical.models import CanonicalDocument, CanonicalQuery


REPO_ROOT = Path(__file__).resolve().parent.parent
SMOKE = REPO_ROOT / "outputs" / "_smoke"


class MockEmbedder:
    def __init__(self, dim: int = 64) -> None:
        self._dim = dim
        self._vocab: dict[str, int] = {}

    @property
    def name(self) -> str:
        return "smoke::mock-embedder"

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self._dim
        for tok in text.lower().split():
            if tok not in self._vocab:
                self._vocab[tok] = len(self._vocab) % self._dim
            v[self._vocab[tok]] += 1.0
        n = sum(x * x for x in v) ** 0.5 or 1.0
        return [x / n for x in v]

    def encode_documents(self, texts):
        return [self._vec(t) for t in texts]

    def encode_query(self, text):
        return self._vec(text)


class MockReader:
    @property
    def name(self) -> str:
        return "smoke::mock-reader"

    def generate(self, prompt: str) -> str:
        return "MOCK_ANSWER"


DOCS = [
    {"doc_id": "d1", "dataset": "smoke",
     "text_raw": "The Eiffel Tower is in Paris, France. It was completed in 1889.",
     "text_normalized": "The Eiffel Tower is in Paris, France. It was completed in 1889."},
    {"doc_id": "d2", "dataset": "smoke",
     "text_raw": "The Louvre Museum is the largest art museum, in Paris.",
     "text_normalized": "The Louvre Museum is the largest art museum, in Paris."},
    {"doc_id": "d3", "dataset": "smoke",
     "text_raw": "The Colosseum is an ancient amphitheater in Rome, Italy.",
     "text_normalized": "The Colosseum is an ancient amphitheater in Rome, Italy."},
]
QUERIES = [
    {"query_id": "q1", "dataset": "smoke", "question": "Where is the Eiffel Tower?"},
    {"query_id": "q2", "dataset": "smoke", "question": "Which is the largest art museum?"},
]


def step(name: str) -> None:
    print(f"\n=== {name} ===")


def main() -> int:
    SMOKE.mkdir(parents=True, exist_ok=True)

    # --- 0. Diagnostic ---
    step("0. Diagnostic")
    for m in ["torch", "transformers", "FlagEmbedding", "openai", "google.generativeai"]:
        found = importlib.util.find_spec(m) is not None
        print(f"  {m}: {'YES' if found else 'NO'}")
    bge_m3 = Path("D:/rag_work/hf_cache/models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181")
    print(f"  BGE-M3 local: {'YES' if bge_m3.exists() else 'NO'} ({bge_m3})")
    print("  (Mock components used for the rest — not native-fidelity)")

    # --- 1. Tiny dataset ---
    step("1. Write tiny dataset")
    docs_path = SMOKE / "docs.jsonl"
    queries_path = SMOKE / "queries.jsonl"
    with docs_path.open("w", encoding="utf-8") as f:
        for d in DOCS:
            f.write(json.dumps(d) + "\n")
    with queries_path.open("w", encoding="utf-8") as f:
        for q in QUERIES:
            f.write(json.dumps(q) + "\n")
    print(f"  docs.jsonl    -> {len(DOCS)} docs")
    print(f"  queries.jsonl -> {len(QUERIES)} queries")

    docs = [CanonicalDocument(**d) for d in DOCS]
    queries = [CanonicalQuery(**q) for q in QUERIES]

    # --- 2. Native E2E smoke ---
    step("2. Native E2E smoke (mock embedder + mock reader)")
    cfg_native = BaselineConfig(
        name="longrag", mode=RunMode.NATIVE, top_k=3,
        extra={"grouping_policy": "one_document_per_unit",
               "target_unit_tokens": 4096, "k1": 100},
    )
    native_idx = SMOKE / "native_idx"
    baseline = LongRAGBaseline(config=cfg_native, work_dir=native_idx)
    baseline._native_embedder = MockEmbedder()  # offline only
    baseline.build_index(docs)
    for fname in ["unit_vectors.npy", "unit_records.json", "index_meta.json"]:
        assert (native_idx / fname).exists(), f"missing {fname}"
    print(f"  index artifacts OK ({native_idx})")

    runner = E2ERunner(
        baseline=baseline, reader=MockReader(),
        prompt=load_prompt("nq"), max_context_chars=10_000,
    )
    results = runner.run_batch(queries)
    assert len(results) == 2
    for r in results:
        assert r.answer == "MOCK_ANSWER"
        assert r.query_id in {"q1", "q2"}
        assert len(r.context_sha256) == 64
        assert len(r.prompt_sha256) == 64
        print(f"    {r.query_id}: answer={r.answer!r} units={r.num_unique_units}")

    manifest_path = SMOKE / "native" / "manifest.json"
    runner.write_manifest(results, manifest_path, extra_meta={"smoke": True})
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["num_queries"] == 2
    assert manifest["baseline"]["name"] == "longrag"
    print(f"  manifest OK ({manifest_path.stat().st_size} bytes)")

    # --- 3. Controlled smoke ---
    step("3. Controlled smoke (injected embedder)")
    cfg_ctrl = BaselineConfig(
        name="longrag", mode=RunMode.CONTROLLED, top_k=3,
        extra={"grouping_policy": "adjacent_document_segments",
               "target_unit_tokens": 50, "k1": 100},
    )
    ctrl_idx = SMOKE / "ctrl_idx"
    embedder = MockEmbedder()
    baseline_ctrl = LongRAGBaseline(
        config=cfg_ctrl, work_dir=ctrl_idx, embedder=embedder,
    )
    baseline_ctrl.build_index(docs)
    print(f"  index built with injected embedder: {embedder.name}")

    for q in queries:
        hits = baseline_ctrl.retrieve(q.question, top_k=3)
        ranks = [r.rank for r in hits]
        assert ranks == list(range(1, len(hits) + 1)), f"bad ranks {ranks}"
        for h in hits:
            assert h.doc_id in {d.doc_id for d in docs}
        print(f"    {q.query_id}: {len(hits)} hits, ranks={ranks}, "
              f"top1_doc={hits[0].doc_id}, top1_score={hits[0].score:.3f}")

    # --- 4. CLI dry-run ---
    step("4. CLI dry-run")
    reader_cfg = SMOKE / "reader_mock.yaml"
    reader_cfg.write_text(
        "reader:\n  type: local\n  model: Qwen/Qwen2.5-0.5B-Instruct\n  device: cpu\n",
        encoding="utf-8",
    )
    rc = cli_main([
        "--docs", str(docs_path),
        "--queries", str(queries_path),
        "--baseline-config",
            str(REPO_ROOT / "baselines" / "longrag_v2" / "config_native.yaml"),
        "--reader-config", str(reader_cfg),
        "--prompt", "nq",
        "--out", str(SMOKE / "cli_out"),
        "--dry-run",
    ])
    assert rc == 0, f"CLI dry-run returned {rc}"
    print(f"  CLI dry-run OK (rc=0)")

    step("DONE")
    print("\nAll smoke checks passed.")
    print(f"Artifacts under: {SMOKE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())