"""FRAMES-PDF v2: render from live Wikipedia HTML (Playwright + Chromium).

Faithful to paper: "We rendered each Wikipedia article into PDF using a
Chromium browser and the Playwright API."

3 concurrent workers, resume support, per-article logging.
Output: D:\rag_work\raw_datasets\frames\pdfs\
"""
from __future__ import annotations

import json
import sys
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from playwright.sync_api import sync_playwright

FRAMES = Path(r"D:\rag_work\raw_datasets\frames")
ARTICLES = FRAMES / "articles.jsonl"
OUT_DIR = FRAMES / "pdfs"
N_WORKERS = 3


def load_titles() -> list[str]:
    titles = []
    with ARTICLES.open(encoding="utf-8") as f:
        for line in f:
            a = json.loads(line)
            t = a.get("title")
            if t:
                titles.append(t)
    return titles


def safe_filename(title: str) -> str:
    safe = "".join(c if c.isalnum() or c in "-_ " else "_" for c in title)
    safe = safe.strip().replace(" ", "_")[:100]
    return safe or "untitled"


def render_chunk(chunk: list[tuple[str, Path]], worker_id: int, log_prefix: str) -> dict:
    """Render a list of (title, out_path) in one Playwright instance."""
    stats = {"ok": 0, "skip": 0, "fail": 0, "failures": []}
    t0 = time.time()

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36 SAH-RAG-Research/1.0"
            ),
            viewport={"width": 1280, "height": 900},
        )
        page = context.new_page()

        for i, (title, out_path) in enumerate(chunk, 1):
            if out_path.exists() and out_path.stat().st_size > 2048:
                stats["skip"] += 1
                continue
            url = f"https://en.wikipedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}"
            try:
                page.goto(url, wait_until="domcontentloaded", timeout=60000)
                # Let images and CSS start loading (best-effort)
                try:
                    page.wait_for_load_state("networkidle", timeout=5000)
                except Exception:
                    pass
                page.wait_for_timeout(500)
                page.pdf(
                    path=str(out_path),
                    format="A4",
                    print_background=True,
                    prefer_css_page_size=False,
                    margin={"top": "0.8cm", "bottom": "0.8cm", "left": "0.8cm", "right": "0.8cm"},
                )
                stats["ok"] += 1
            except Exception as e:
                stats["fail"] += 1
                stats["failures"].append((title, str(e)[:200]))
            if i % 20 == 0:
                elapsed = time.time() - t0
                rate = (stats["ok"] + stats["fail"]) / max(elapsed, 0.001)
                print(f"[W{worker_id} {i}/{len(chunk)}] ok={stats['ok']} skip={stats['skip']} fail={stats['fail']} {rate:.2f}/s", flush=True)

        context.close()
        browser.close()

    return stats


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    titles = load_titles()
    print(f"[OK] Loaded {len(titles)} titles", flush=True)

    # Build work items
    items = []
    seen = set()
    for t in titles:
        fname = safe_filename(t) + ".pdf"
        if fname in seen:
            continue
        seen.add(fname)
        items.append((t, OUT_DIR / fname))

    print(f"[OK] Unique work items: {len(items)}", flush=True)

    # Split into N_WORKERS chunks (round-robin to balance long/short articles)
    chunks = [[] for _ in range(N_WORKERS)]
    for i, item in enumerate(items):
        chunks[i % N_WORKERS].append(item)

    print(f"[OK] Starting {N_WORKERS} workers", flush=True)
    t0 = time.time()
    total = {"ok": 0, "skip": 0, "fail": 0, "failures": []}

    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        futures = [
            ex.submit(render_chunk, chunk, i + 1, f"W{i+1}")
            for i, chunk in enumerate(chunks)
        ]
        for fut in as_completed(futures):
            stats = fut.result()
            for k in ("ok", "skip", "fail"):
                total[k] += stats[k]
            total["failures"].extend(stats["failures"])

    elapsed = time.time() - t0
    print(f"\n[DONE] {elapsed/60:.1f} min total", flush=True)
    print(f"  ok={total['ok']} skip={total['skip']} fail={total['fail']}", flush=True)
    if total["failures"]:
        print(f"\n[FAILURES] {len(total['failures'])}:", flush=True)
        for title, err in total["failures"][:10]:
            print(f"  - {title}: {err}", flush=True)

    # Disk usage
    total_size = sum(p.stat().st_size for p in OUT_DIR.glob("*.pdf"))
    print(f"\n[OUT] {OUT_DIR}", flush=True)
    print(f"[SIZE] {total_size / (1024*1024):.1f} MB across {len(list(OUT_DIR.glob('*.pdf')))} PDFs", flush=True)

    return 0 if total["fail"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
