# FRAMES-PDF rendering (Wikipedia live HTML)

Reproduces the paper's data preparation: "We rendered each Wikipedia article into PDF using a Chromium browser and the Playwright API."

## Requirements

- Python 3.12
- pip install playwright
- python -m playwright install chromium

## Output

- Destination: `D:\rag_work\raw_datasets\frames\pdfs\`
- 2,521 PDFs (one per FRAMES article)
- Total size: ~1.9 GB
- Average headers/PDF: ~23 (h2/h3 sections, tables, figures preserved)

## Usage

    python scripts\frames_render\render_from_wikipedia.py

Resume support: skips PDFs that already exist and are larger than 2 KB.

## Previous attempt (v1, deprecated)

An earlier version parsed the raw wikitext from `articles.jsonl`, which
contains no HTML structure (no headers, no tables, no images). Those PDFs
are in `D:\rag_work\raw_datasets\frames\pdfs_v1_plaintext\` and must
NOT be used for native HiKEY evaluation.

## Verification

After rendering, verify structure:

    python -c "from workspace.hikey_native_dev.components.dhp_parser import parse_pdf, BLOCK_HEADER; from pathlib import Path; import random; pdfs = sorted(Path(r'D:\rag_work\raw_datasets\frames\pdfs').glob('*.pdf')); [print(p.name, len([b for b in parse_pdf(p) if b.block_type == BLOCK_HEADER])) for p in random.sample(pdfs, 5)]"

Expected: each PDF has > 0 headers.