"""Smoke test: two-step pipeline with real BGE-M3 + mock reader.

Controlled mode (not native: native requires BGE-large-en-v1.5 + API key).
Uses real BGE-M3 embedder to verify that the two-step runner works
end-to-end with a real embedding stack.
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SMOKE_DIR = REPO_ROOT / "outputs" / "_smoke_two_step"
BGE_M3_PATH = Path(
    "D:/rag_work/hf_cache/models--BAAI--bge-m3/snapshots/"
    "5617a9f61b028005a4858fdac845db406aefb181"
)

from baselines.base import BaselineConfig, RunMode
from baselines.longrag_v2.adapter import LongRAGBaseline
from baselines.longrag_v2.native_e2e import (
    E2ERunner,
    load_prompt,
    single_ans_em,
    has_correct_answer,
)
from sah_rag.canonical.models import CanonicalDocument


class BGE_M3_Embedder:
    """Real BGE-M3 embedder via transformers, CLS pooling."""

    def __init__(self, model_path: Path) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        self._model = AutoModel.from_pretrained(str(model_path))
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = self._model.to(self._device)
        self._model.eval()

    @property
    def name(self) -> str:
        return "BAAI/bge-m3"

    def encode_documents(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        batch_size = 4
        with self._torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                enc = self._tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=8192,
                    return_tensors="pt",
                ).to(self._device)
                hidden = self._model(**enc).last_hidden_state
                cls = hidden[:, 0]
                cls = self._torch.nn.functional.normalize(cls, p=2, dim=-1)
                out.extend(cls.cpu().numpy().tolist())
        return out

    def encode_query(self, text: str) -> list[float]:
        return self.encode_documents([text])[0]


class MockReader:
    """Returns a deterministic answer per query.

    Call order (2 calls per query):
        call 1 -> step 1 for query 1 -> long answer mentioning gold
        call 2 -> step 2 for query 1 -> gold answer
        call 3 -> step 1 for query 2 -> long answer mentioning gold
        call 4 -> step 2 for query 2 -> gold answer
    """

    ANSWERS = ["Paris", "Louvre"]  # one per query, in order

    def __init__(self) -> None:
        self.calls: list[str] = []

    @property
    def name(self) -> str:
        return "smoke::mock-reader"

    def generate(self, prompt: str) -> str:
        self.calls.append(prompt)
        n = len(self.calls)
        q_idx = (n - 1) // 2  # 0 for calls 1-2, 1 for calls 3-4
        if q_idx >= len(self.ANSWERS):
            return ""
        gold = self.ANSWERS[q_idx]
        is_step_1 = (n % 2 == 1)
        if is_step_1:
            return f"The {gold} is mentioned. Answer: {gold}"
        return gold


DOCS = [
    {
        "doc_id": "d1",
        "dataset": "smoke",
        "text_raw": "The Eiffel Tower is in Paris, France.",
        "text_normalized": "The Eiffel Tower is in Paris, France.",
        "title": "Eiffel Tower",
    },
    {
        "doc_id": "d2",
        "dataset": "smoke",
        "text_raw": "The Louvre Museum is in Paris.",
        "text_normalized": "The Louvre Museum is in Paris.",
        "title": "Louvre",
    },
    {
        "doc_id": "d3",
        "dataset": "smoke",
        "text_raw": "The Colosseum is in Rome, Italy.",
        "text_normalized": "The Colosseum is in Rome, Italy.",
        "title": "Colosseum",
    },
]

QUERIES = [
    {"query_id": "q1", "question": "Where is the Eiffel Tower?", "answers": ["Paris"]},
    {"query_id": "q2", "question": "Which museum is in Paris?", "answers": ["Louvre"]},
]


def main() -> int:
    SMOKE_DIR.mkdir(parents=True, exist_ok=True)

    print("\n=== 1. Environment ===")
    print(f"  BGE-M3 path: {BGE_M3_PATH}")
    print(f"  exists: {BGE_M3_PATH.exists()}")
    if not BGE_M3_PATH.exists():
        print("  FATAL: BGE-M3 not found")
        return 1

    print("\n=== 2. Load BGE-M3 (may take 20-30s) ===")
    t0 = time.perf_counter()
    embedder = BGE_M3_Embedder(BGE_M3_PATH)
    print(f"  Loaded in {time.perf_counter() - t0:.1f}s")
    print(f"  Device: {embedder._device}")

    print("\n=== 3. Build corpus ===")
    docs = [
        CanonicalDocument(
            doc_id=d["doc_id"],
            dataset=d["dataset"],
            text_raw=d["text_raw"],
            text_normalized=d["text_normalized"],
            metadata={"title": d["title"]},
        )
        for d in DOCS
    ]
    print(f"  {len(docs)} documents")

    print("\n=== 4. Build index (controlled mode, injected BGE-M3) ===")
    cfg = BaselineConfig(
        name="longrag",
        mode=RunMode.CONTROLLED,
        top_k=3,
        extra={
            "grouping_policy": "one_document_per_unit",
            "target_unit_tokens": 4096,
            "k1": 100,
        },
    )
    baseline = LongRAGBaseline(
        config=cfg,
        work_dir=SMOKE_DIR / "index",
        embedder=embedder,
    )
    t0 = time.perf_counter()
    baseline.build_index(docs)
    print(f"  Index built in {time.perf_counter() - t0:.1f}s")
    print(f"  Metadata: {json.loads((SMOKE_DIR / 'index' / 'index_meta.json').read_text())}")

    print("\n=== 5. Two-step E2E run ===")
    reader = MockReader()
    runner = E2ERunner(
        baseline=baseline,
        reader=reader,
        main_prompt=load_prompt("nq"),
        extraction_prompt=load_prompt("answer_extraction"),
        dataset_name="nq",
    )
    queries_tuples = [(q["query_id"], q["question"]) for q in QUERIES]
    t0 = time.perf_counter()
    results = runner.run_batch(queries_tuples)
    print(f"  Completed in {time.perf_counter() - t0:.1f}s")
    print(f"  Reader calls: {len(reader.calls)} (expected {2 * len(QUERIES)})")

    print("\n=== 6. Verify results ===")
    all_ok = True
    for r, q in zip(results, QUERIES):
        assert r.query_id == q["query_id"]
        assert len(r.main_prompt_sha256) == 64
        assert len(r.extraction_prompt_sha256) == 64
        assert len(r.context_sha256) == 64
        assert len(r.titles) > 0

        em = single_ans_em(r.short_answer, q["answers"])
        sub = has_correct_answer(r.long_answer, q["answers"])
        print(f"  {r.query_id}: query={r.query!r}")
        print(f"           long_answer={r.long_answer!r}")
        print(f"           short_answer={r.short_answer!r}")
        print(f"           titles={r.titles}")
        print(f"           top1_doc={r.retrieved[0].doc_id if r.retrieved else None}")
        print(f"           em={em}, substring={sub}")
        if em != 1:
            all_ok = False

    print("\n=== 7. Write manifest ===")
    manifest_path = SMOKE_DIR / "manifest.json"
    runner.write_manifest(results, manifest_path)
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    print(f"  Manifest size: {manifest_path.stat().st_size} bytes")
    print(f"  schema_version: {data['schema_version']}")
    print(f"  num_queries: {data['num_queries']}")
    print(f"  main_prompt: {data['main_prompt']['name']}")
    print(f"  extraction_prompt: {data['extraction_prompt']['name']}")
    print(f"  demos_chars: {data['demos']['length_chars']}")

    print("\n=== Summary ===")
    print(f"  All metric checks passed: {all_ok}")
    print(f"  Reader called exactly twice per query: {len(reader.calls) == 2 * len(QUERIES)}")
    print(f"  Artifacts under: {SMOKE_DIR}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(2)