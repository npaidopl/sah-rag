"""Prepare FRAMES queries + matched PDF corpus for native HiKEY runs.

Parses test.tsv, converts wikipedia_link_N URLs to titles, copies
matched PDFs from the rendered corpus.

Usage:
    python scripts/frames_render/prepare_smoke.py --limit 20
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import urllib.parse
from pathlib import Path


def url_to_title(url: str) -> str:
    """Extract article title from Wikipedia URL."""
    url = url.strip().split("#")[0]
    for prefix in (
        "https://en.wikipedia.org/wiki/",
        "http://en.wikipedia.org/wiki/",
        "https://en.wikipedia.org/w/index.php?title=",
    ):
        if url.startswith(prefix):
            url = url[len(prefix):]
            break
    return urllib.parse.unquote(url).replace("_", " ").strip()


def safe_filename(title: str) -> str:
    safe = "".join(c if c.isalnum() or c in "-_ " else "_" for c in title)
    return safe.strip().replace(" ", "_")[:100]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--frames-root", default=r"D:\rag_work\raw_datasets\frames")
    ap.add_argument("--corpus-out", default=r"C:\Users\USER1\Documents\SAH-RAG\workspace\hikey_smoke_test\frames_corpus")
    ap.add_argument("--queries-out", default=r"C:\Users\USER1\Documents\SAH-RAG\workspace\hikey_smoke_test\frames_queries.json")
    args = ap.parse_args()

    frames = Path(args.frames_root)
    pdfs = frames / "pdfs"
    test_tsv = frames / "test.tsv"
    corpus_out = Path(args.corpus_out)
    queries_out = Path(args.queries_out)

    if corpus_out.exists():
        shutil.rmtree(corpus_out)
    corpus_out.mkdir(parents=True, exist_ok=True)

    link_cols = ["wikipedia_link_1"] + [f"wikipedia_link_{i}" for i in range(2, 11)] + ["wikipedia_link_11+"]

    queries = []
    titles_needed: set[str] = set()

    with test_tsv.open(encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for i, row in enumerate(reader):
            if i >= args.limit:
                break
            q = (row.get("Prompt") or "").strip()
            answer = (row.get("Answer") or "").strip()
            links = []
            for col in link_cols:
                v = (row.get(col) or "").strip()
                if v and v.startswith("http"):
                    t = url_to_title(v)
                    if t:
                        links.append(t)
                        titles_needed.add(t)
            queries.append({"query_id": f"f{i+1}", "question": q, "answer": answer, "gold_articles": links})

    copied = 0
    missing = []
    for title in titles_needed:
        src = pdfs / f"{safe_filename(title)}.pdf"
        if src.exists():
            shutil.copy2(src, corpus_out / src.name)
            copied += 1
        else:
            missing.append(title)

    queries_out.write_text(json.dumps(queries, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[OK] {len(queries)} queries, {len(titles_needed)} gold titles, {copied} PDFs matched")
    if missing:
        print(f"[WARN] Missing {len(missing)} PDFs (first 5): {missing[:5]}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())