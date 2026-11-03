"""CLI entry point for the native HiKEY pipeline."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .pipeline import HiKEYNativePipeline


def main() -> int:
    ap = argparse.ArgumentParser(description="HiKEY native pipeline CLI")
    ap.add_argument("--corpus", required=True, help="Directory of PDFs")
    ap.add_argument("--queries", required=True, help="JSON file: [{\"query_id\": str, \"question\": str}, ...]")
    ap.add_argument("--out", required=True, help="Output JSONL path")
    ap.add_argument("--work-dir", default="./hikey_native_work")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--mock", action="store_true", help="Use mock components (no model downloads)")
    ap.add_argument("--mock-encoders", action="store_true", help="Real PDF parser, mock encoders")
    ap.add_argument("--answer", action="store_true", help="Also generate answers with the LVLM reader")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.mock_encoders and not args.mock:
        pipeline = HiKEYNativePipeline(
            work_dir=args.work_dir, mock=True, mock_parser=False,
        )
    else:
        pipeline = HiKEYNativePipeline(work_dir=args.work_dir, mock=args.mock)
    pipeline.index(Path(args.corpus))

    queries = json.loads(Path(args.queries).read_text(encoding="utf-8"))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for q in queries:
            qid = q.get("query_id") or q.get("id") or "q"
            question = q["question"]
            if args.answer:
                result = pipeline.answer(question, top_k=args.top_k)
            else:
                result = {"query_id": qid, "question": question, "chunks": pipeline.retrieve(question, top_k=args.top_k)}
            result["query_id"] = qid
            # Preserve optional fields from the query JSON (gold_articles, answer, etc.)
            for extra_key in ("gold_articles", "answer", "reference_answers"):
                if extra_key in q:
                    result[extra_key] = q[extra_key]
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            print(f"[{qid}] {len(result.get('chunks', []))} chunks", file=sys.stderr)

    print(f"Wrote {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())