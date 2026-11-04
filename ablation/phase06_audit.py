import hashlib
import inspect
import itertools
import re
from statistics import median

from sah_rag.parsing import headings as headings_mod
from sah_rag.parsing.deterministic import DeterministicStructuralParser
from sah_rag.chunking.hierarchical import HierarchicalChildChunker
from sah_rag.chunking.config import ChildChunkerConfig
from sah_rag.chunking.tokenization import RegexOffsetTokenizer

from ablation.config import ALL_DATASETS
from ablation.loaders import iter_canonical_documents_local
from ablation.chunkers.variants import (
    ParagraphParentsParser, PlaceboBoundariesParser,
)
from ablation.chunkers.boundaries_disabled import BoundariesDisabledChunker


def sha_short(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def audit_heading_module():
    print("=" * 78)
    print("### HEADING MODULE — Greek keyword probe")
    print("=" * 78)
    src = inspect.getsource(headings_mod)
    # Find any Greek-looking string literals
    greek_tokens = set(re.findall(r"[\u0370-\u03ff\u1f00-\u1fff]{3,}", src))
    print(f"Greek tokens found in headings.py: {sorted(greek_tokens)}")
    # Look for keyword list
    for line in src.splitlines():
        if any(kw in line for kw in ["chapter", "section", "article", "Άρθρο", "ΜΕΡΟΣ", "ΚΕΦΑΛΑΙΟ"]):
            print(f"  > {line.strip()[:120]}")
    print()


def audit_dataset(ds: str, max_docs: int = 5) -> dict:
    try:
        docs = list(itertools.islice(iter_canonical_documents_local(ds), max_docs))
    except Exception as e:
        return {"error": str(e)}
    if not docs:
        return {"error": "empty"}

    text_hashes = [sha_short(d.text_normalized) for d in docs]
    unique_texts = len(set(text_hashes))

    real_parser = DeterministicStructuralParser()
    para_parser = ParagraphParentsParser()
    place_parser = PlaceboBoundariesParser(seed=42)

    real_parents, para_parents, place_parents = [], [], []
    for d in docs:
        real_parents.append(len(real_parser.parse(d).parents))
        para_parents.append(len(para_parser.parse(d).parents))
        place_parents.append(len(place_parser.parse(d).parents))

    a6_cross = 0
    a4_chunks = 0
    a6_chunks = 0
    std_chunker = None
    a6_chunker = None

    if max(real_parents) > 1:
        tok = RegexOffsetTokenizer()
        cfg = ChildChunkerConfig(chunk_size=256, chunk_overlap=32)
        std_chunker = HierarchicalChildChunker(tokenizer=tok, config=cfg)
        a6_chunker = BoundariesDisabledChunker(tokenizer=tok, config=cfg)
        for d in docs:
            r = real_parser.parse(d)
            a4c = std_chunker.chunk_document(r)
            a6c = a6_chunker.chunk_document_with_doc_text(d, r)
            a4_chunks += len(a4c)
            a6_chunks += len(a6c)
            a6_cross += sum(1 for c in a6c if c.metadata.get("a6_crosses_parent_boundary"))

    first_text = docs[0].text_normalized[:5000]
    probes = {
        "gr_arthro":    len(re.findall(r"Άρθρο\s*\d+", first_text)),
        "gr_meros":     len(re.findall(r"\bΜΕΡΟΣ\b", first_text)),
        "gr_kefalaio":  len(re.findall(r"\bΚΕΦΑΛΑΙΟ\b", first_text)),
        "gr_paragr":    len(re.findall(r"^Παράγραφος\b", first_text, re.MULTILINE)),
        "article":      len(re.findall(r"\bArticle\s+\d+", first_text)),
        "section":      len(re.findall(r"\bSection\s+\d+", first_text)),
        "numbered":     len(re.findall(r"^\d+\.\s+", first_text, re.MULTILINE)),
        "uppercase":    len([l for l in first_text.split("\n")[:80] if l.strip() and l.strip().isupper()]),
    }

    return {
        "n_docs": len(docs),
        "unique_texts": unique_texts,
        "real_par_med": median(real_parents),
        "real_par_min": min(real_parents),
        "real_par_max": max(real_parents),
        "para_par_med": median(para_parents),
        "place_par_med": median(place_parents),
        "a4_chunks": a4_chunks,
        "a6_chunks": a6_chunks,
        "a6_cross": a6_cross,
        "probes": probes,
        "sample": docs[0].text_normalized[:250].replace("\n", " "),
    }


def main():
    audit_heading_module()

    print("=" * 100)
    print("### STRUCTURE AUDIT — 17 datasets × 5 docs")
    print("=" * 100)
    print()

    hdr = (f"{'dataset':<20} {'n':>2} {'uniq':>4} "
           f"{'real_par':>8} {'para_par':>8} {'place_par':>9} "
           f"{'a4_chk':>6} {'a6_chk':>6} {'a6_X':>4}  verdict")
    print(hdr)
    print("-" * len(hdr))

    verdicts = {}
    for ds in ALL_DATASETS:
        r = audit_dataset(ds, 5)
        if "error" in r:
            print(f"{ds:<20} ERROR: {r['error']}")
            continue
        verdicts[ds] = r

        v = []
        if r["real_par_med"] <= 1:
            v.append("FLAT")
        if r["unique_texts"] == 1:
            v.append("DEGEN_TXT")
        elif r["unique_texts"] < r["n_docs"]:
            v.append(f"DUP({r['unique_texts']}/{r['n_docs']})")
        if r["real_par_med"] > 1 and r.get("a6_cross", 0) == 0:
            v.append("A6_no_effect")
        if r["real_par_med"] > 1 and r["a6_cross"] > 0:
            v.append("A6_ok")

        print(f"{ds:<20} {r['n_docs']:>2} {r['unique_texts']:>4} "
              f"{r['real_par_med']:>8.0f} {r['para_par_med']:>8.0f} {r['place_par_med']:>9.0f} "
              f"{r['a4_chunks']:>6} {r['a6_chunks']:>6} {r['a6_cross']:>4}  "
              f"{','.join(v) if v else 'OK'}")

    print()
    print("=" * 100)
    print("### HEADING PROBES (first 5000 chars)")
    print("=" * 100)
    print()
    hdr2 = (f"{'dataset':<20} "
            f"{'gr_arth':>7} {'gr_mer':>6} {'gr_kef':>6} {'gr_par':>6} "
            f"{'artic':>6} {'sect':>5} {'num':>4} {'upper':>6}")
    print(hdr2)
    print("-" * len(hdr2))
    for ds, r in verdicts.items():
        p = r["probes"]
        print(f"{ds:<20} "
              f"{p['gr_arthro']:>7} {p['gr_meros']:>6} {p['gr_kefalaio']:>6} {p['gr_paragr']:>6} "
              f"{p['article']:>6} {p['section']:>5} {p['numbered']:>4} {p['uppercase']:>6}")

    print()
    print("=" * 100)
    print("### SAMPLE TEXT (first 250 chars) — datasets with FLAT")
    print("=" * 100)
    print()
    for ds, r in verdicts.items():
        if r["real_par_med"] <= 1:
            print(f"[{ds}] {r['sample']}")
            print()


if __name__ == "__main__":
    raise SystemExit(main())