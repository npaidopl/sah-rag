"""Render FRAMES articles to PDF (Playwright + Chromium).

Produces FRAMES-PDF matching the paper's rendering approach:
  1. Generate HTML from article text (title + body)
  2. Render HTML to PDF via headless Chromium
  3. Save to pdfs/ folder

Usage:
    python _render_frames.py --limit 20        # test
    python _render_frames.py                   # full (2,521)
"""
from __future__ import annotations

import argparse
import html
import json
import sys
import time
from pathlib import Path


def article_to_html(article: dict) -> str:
    title = html.escape(article.get("title", "Untitled"))
    text = article.get("text", "")
    wiki = html.escape(article.get("wiki", ""))
    # Text is wikitext-ish: preserve line breaks, strip simple templates
    body = html.escape(text)
    # Basic cleanup: remove {{...}} templates
    import re
    body = re.sub(r"\{\{[^{}]*\}\}", "", body)
    body = re.sub(r"\[\[([^\]|]+)\|([^\]]+)\]\]", r"\2", body)
    body = re.sub(r"\[\[([^\]]+)\]\]", r"\1", body)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
  body {{ font-family: Georgia, serif; margin: 2cm; line-height: 1.5; font-size: 11pt; }}
  h1 {{ font-size: 22pt; border-bottom: 1px solid #ccc; padding-bottom: 6pt; }}
  .meta {{ color: #666; font-size: 9pt; margin-bottom: 12pt; }}
  pre {{ white-space: pre-wrap; font-family: Georgia, serif; font-size: 11pt; }}
</style>
</head>
<body>
<h1>{title}</h1>
<div class="meta">Wikipedia article · {wiki}</div>
<pre>{body}</pre>
</body>
</html>"""


def render_one(page, article: dict, out_path: Path) -> bool:
    try:
        html_content = article_to_html(article)
        page.set_content(html_content, wait_until="domcontentloaded")
        page.pdf(
            path=str(out_path),
            format="A4",
            print_background=True,
            margin={"top": "1cm", "bottom": "1cm", "left": "1cm", "right": "1cm"},
        )
        return True
    except Exception as e:
        print(f"[FAIL] {article.get('title', '?')[:50]}: {e}", file=sys.stderr)
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--articles", default=r"D:\rag_work\raw_datasets\frames\articles.jsonl")
    ap.add_argument("--out-dir", default=r"D:\rag_work\raw_datasets\frames\pdfs")
    args = ap.parse_args()

    articles_path = Path(args.articles)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load articles
    articles: list[dict] = []
    with articles_path.open(encoding="utf-8") as f:
        for line in f:
            articles.append(json.loads(line))
    if args.limit:
        articles = articles[:args.limit]
    print(f"[OK] Loaded {len(articles)} articles")

    # Render with Playwright
    from playwright.sync_api import sync_playwright

    start = time.time()
    ok = 0
    fail = 0
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        page = browser.new_page()
        for i, article in enumerate(articles, 1):
            key = article.get("key") or article.get("page_id") or f"article_{i}"
            safe_key = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(key))[:80]
            out_path = out_dir / f"{safe_key}.pdf"
            if out_path.exists() and out_path.stat().st_size > 1024:
                ok += 1
                continue
            if render_one(page, article, out_path):
                ok += 1
            else:
                fail += 1
            if i % 10 == 0 or i == len(articles):
                elapsed = time.time() - start
                rate = i / max(elapsed, 0.001)
                eta = (len(articles) - i) / max(rate, 0.001)
                print(f"[{i}/{len(articles)}] ok={ok} fail={fail} "
                      f"{rate:.2f} art/s, ETA {eta/60:.1f} min")
        browser.close()

    elapsed = time.time() - start
    print(f"\n[DONE] {ok} ok, {fail} fail in {elapsed/60:.1f} min")
    print(f"[OUT] {out_dir}")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
