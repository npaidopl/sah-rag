import sys, io, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np

from ablation.loaders import iter_canonical_documents_local
from ablation.chunkers.variants import FlatHybridParser
from sah_rag.parsing.deterministic import DeterministicStructuralParser
from sah_rag.chunking.hierarchical import HierarchicalChildChunker
from sah_rag.chunking.config import ChildChunkerConfig
from sah_rag.chunking.tokenization import RegexOffsetTokenizer
from ablation.embedding.bge_m3 import BGEM3Embedder
from ablation.embedding.cache import EmbeddingCache


def main() -> int:
    dataset = "clerk"
    N_DOCS, MAX_CHUNKS = 2, 10
    variant = "A4_full_sah"

    print(f"Loading {N_DOCS} docs from {dataset}")
    docs = []
    for d in iter_canonical_documents_local(dataset):
        docs.append(d)
        if len(docs) >= N_DOCS:
            break
    print(f"  loaded {len(docs)}")

    parser = DeterministicStructuralParser()
    tok = RegexOffsetTokenizer()
    cfg = ChildChunkerConfig(chunk_size=256, chunk_overlap=32)
    chunker = HierarchicalChildChunker(tokenizer=tok, config=cfg)

    chunks = []
    for d in docs:
        parsed = parser.parse(d)
        chunks.extend(chunker.chunk_document(parsed))
        if len(chunks) >= MAX_CHUNKS:
            break
    chunks = chunks[:MAX_CHUNKS]
    print(f"  produced {len(chunks)} chunks")
    for c in chunks[:3]:
        print(f"    {c.chunk_id}  tokens={c.token_count}")

    print("\nLoading BGE-M3")
    embedder = BGEM3Embedder(max_length=512, use_fp16=False, device="cpu")
    print(f"  load_time: {embedder.load_time_s:.1f}s")
    print(f"  device: {embedder.device}  dense_dim: {embedder.dense_dim}")
    print(f"  snapshot: {embedder.snapshot_path}")

    print(f"\nEncoding {len(chunks)} chunks ...")
    t0 = time.time()
    dense, sparse = embedder.encode_hybrid([c.text_normalized for c in chunks])
    dt = time.time() - t0
    print(f"  elapsed: {dt:.2f}s  ({dt/len(chunks):.3f}s/chunk)")
    print(f"  dense: shape={dense.shape} dtype={dense.dtype}")
    norms = np.linalg.norm(dense, axis=1)
    print(f"  dense norms: min={norms.min():.3f} max={norms.max():.3f} mean={norms.mean():.3f}")
    nnz = [len(s) for s in sparse]
    print(f"  sparse nnz: min={min(nnz)} max={max(nnz)} mean={sum(nnz)/len(nnz):.1f}")

    print(f"\nSaving to cache: variant={variant} dataset={dataset}")
    cache = EmbeddingCache(variant=variant, dataset=dataset)
    cache.save(
        chunk_ids=[c.chunk_id for c in chunks],
        dense=dense,
        sparse=sparse,
        config={"embedder": "BAAI/bge-m3", "max_length": 512, "device": "cpu"},
    )
    print(f"  {cache.root}")
    for p in (cache.dense_path, cache.chunks_path, cache.manifest_path):
        print(f"    {p.name}: {p.stat().st_size} bytes")

    print("\nReloading from cache ...")
    ids2, dense2, sparse2 = cache.load()
    assert len(ids2) == len(chunks)
    assert ids2 == [c.chunk_id for c in chunks], "chunk_id order mismatch"
    assert dense2.shape == dense.shape, "dense shape mismatch"
    assert np.allclose(dense2, dense, atol=1e-6), "dense values mismatch"
    for i, (a, b) in enumerate(zip(sparse2, sparse)):
        if a != b:
            raise AssertionError(f"sparse mismatch at {i}")
    print("  round-trip: PASSED")

    print("\nSanity: exact chunk from clerk doc[0]")
    doc0_chunks = [c for c in chunks if c.doc_id == docs[0].doc_id]
    if doc0_chunks:
        print(f"  first chunk of doc[0]: {doc0_chunks[0].chunk_id}")
        print(f"  text preview: {doc0_chunks[0].text_normalized[:120]!r}")

    print("\nALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())