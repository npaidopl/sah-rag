"""Preview with GT coverage check. Shows sampling impact per dataset."""
import sys
from pathlib import Path
SAH = Path(r"C:\Users\USER1\Documents\SAH-RAG")
sys.path.insert(0, str(SAH))

from ablation.config import (ALL_DATASETS, STRUCTURE_RICH_DATASETS,
                              sample_limit, sample_docs_gt_aware)
from ablation.loaders import (iter_canonical_documents_local,
                              iter_canonical_examples_local)
from sah_rag.parsing.deterministic import DeterministicStructuralParser
from sah_rag.chunking.hierarchical import HierarchicalChildChunker
from sah_rag.chunking.config import ChildChunkerConfig
from sah_rag.chunking.tokenization import RegexOffsetTokenizer

parser = DeterministicStructuralParser()
tok = RegexOffsetTokenizer()
chunker = HierarchicalChildChunker(tokenizer=tok,
                                    config=ChildChunkerConfig(chunk_size=256, chunk_overlap=32))

print(f"{'dataset':<22} {'docs':>7} {'sample':>7} {'kept':>7} {'GT cov':>8} {'est_chk':>10} {'x2+rich':>9}")
print("-" * 95)

total_a4 = 0
total_tier1 = 0
for ds in ALL_DATASETS:
    limit = sample_limit(ds)
    # Doc count + GT count
    all_docs = []
    gt_set = set()
    for ex in iter_canonical_examples_local(ds):
        gt_set.update(ex.ground_truth.relevant_ids)
    for d in iter_canonical_documents_local(ds):
        all_docs.append(d.doc_id)
    n_docs = len(all_docs)
    gt_in_corpus = gt_set & set(all_docs)

    if limit is None:
        kept_docs = set(all_docs)
        kept = n_docs
    else:
        kept_docs = sample_docs_gt_aware(ds, limit)
        kept = len(kept_docs)

    # GT coverage
    n_queries = 0
    n_queries_gt_in_sample = 0
    for ex in iter_canonical_examples_local(ds):
        n_queries += 1
        if ex.ground_truth.relevant_ids and \
           any(g in kept_docs for g in ex.ground_truth.relevant_ids):
            n_queries_gt_in_sample += 1
    gt_cov = f"{n_queries_gt_in_sample}/{n_queries}"

    # Estimate chunk count: sample 3 kept docs
    sample_chunks = 0
    sample_n = 0
    for d in iter_canonical_documents_local(ds):
        if d.doc_id not in kept_docs:
            continue
        if sample_n >= 3:
            break
        p = parser.parse(d)
        sample_chunks += len(chunker.chunk_document(p))
        sample_n += 1
    if sample_n > 0:
        avg_cpc = sample_chunks / sample_n
        est = int(avg_cpc * kept)
    else:
        avg_cpc = 0
        est = 0

    is_rich = ds in STRUCTURE_RICH_DATASETS
    # A4+A7 universal + (A1+A6+A9 only if rich) = 2x for poor, 5x for rich
    multiplier = 5 if is_rich else 2
    total_tier1 += est * multiplier
    total_a4 += est

    sample_str = str(limit) if limit else "full"
    rich_str = f"×{multiplier}" if is_rich else "×2"
    print(f"{ds:<22} {n_docs:>7} {sample_str:>7} {kept:>7} {gt_cov:>8} {est:>10} {est*multiplier:>9}")

print("-" * 95)
print(f"Total A4 chunks: {total_a4:,}")
print(f"Total Tier-1 embeds: {total_tier1:,}")
print()
print(f"Est GPU time (single T4, 50ms/chunk): {total_tier1 * 0.05 / 3600:.1f}h")
print(f"Est GPU time (4x T4):               {total_tier1 * 0.05 / 3600 / 4:.1f}h")