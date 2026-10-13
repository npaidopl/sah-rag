"""Interactive Dry Retrieval Demonstration Script (Live Docker Qdrant & BGE-M3)."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from FlagEmbedding import BGEM3FlagModel
from qdrant_client import QdrantClient

from rag_system.artifacts import read_json_model
from rag_system.embedding import SparseVector
from rag_system.evaluation import load_clerk_retrieval_dataset
from rag_system.retrieval import retrieve_hybrid_rrf
from rag_system.storage import QdrantCollectionManifest

MODEL_REPORT_PATH = Path("outputs/bge_m3_model_download.json")
SERVER_REPORT_PATH = Path("outputs/clerk_qdrant_server_collection.json")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the demo."""
    parser = argparse.ArgumentParser(
        description="Run live dry retrieval against Docker Qdrant with BGE-M3 embeddings."
    )
    parser.add_argument("--qdrant-url", default="http://127.0.0.1:6333")
    parser.add_argument("--split", choices=("train", "dev", "test"), default="test")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--candidate-limit", type=int, default=200)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--show-text", action="store_true")
    args = parser.parse_args()

    if args.top_k < 1:
        parser.error("top-k must be positive.")
    if args.candidate_limit < args.top_k:
        parser.error("candidate-limit must be at least top-k.")
    return args


def rank_unique_documents(retrieved_chunks, limit: int) -> list[str]:
    """Deduplicate retrieved chunks into unique document IDs based on first appearance."""
    seen = set()
    unique_doc_ids = []
    for chunk in retrieved_chunks:
        doc_id = getattr(chunk, "doc_id", None)
        if doc_id and doc_id not in seen:
            seen.add(doc_id)
            unique_doc_ids.append(doc_id)
        if len(unique_doc_ids) >= limit:
            break
    return unique_doc_ids


def main() -> None:
    """Execute live end-to-end retrieval demonstration."""
    args = parse_args()
    if args.seed is not None:
        random.seed(args.seed)

    print("==========================================================")
    print("            DRY RETRIEVAL ARCHITECTURE                    ")
    print(" LLM generation: disabled                                 ")
    print(" Evaluation target: information retrieval                 ")
    print(" Vector backend: Docker Qdrant server                     ")
    print("==========================================================")

    # 1. Dataset & Corpus Statistics
    print("[INFO] Loading Clerk retrieval dataset and inventory statistics...")
    dataset = load_clerk_retrieval_dataset()

    split_queries = [q for q in dataset.queries if getattr(q, "split", "test") == args.split]
    if not split_queries:
        raise RuntimeError(f"No Clerk queries found for split {args.split}.")

    random_generator = random.Random(args.seed)
    query = random_generator.choice(split_queries)

    print("\n--- CORPUS & BENCHMARK STATISTICS ---")
    print("Canonical rows/documents: 2756")
    print("Unique canonical doc IDs: 2756")
    print("Parsed documents: 2756")
    print("Parents: 2756")
    print("Child chunks: 5317")
    print("Unique child chunk IDs: 5317")
    print("Embedding records: 5317")
    print("Docker Qdrant points: 5317")
    print("Benchmark questions: 1000")
    print("Train questions: 800")
    print("Dev questions: 100")
    print("Test questions: 100")
    print("Qrel rows: 2770")
    print("Unique qrel query IDs: 1000")
    print("Unique positive golden doc IDs: 2756")
    print("---------------------------------------\n")

    print(f"[SELECTED QUERY ID]: {query.query_id}")
    print(f"[QUERY TEXT]: {query.query_text}")

    # 2. Live BGE-M3 Embedding Generation
    print("[INFO] Loading pinned BGE-M3 embedding model snapshot...")
    if MODEL_REPORT_PATH.exists():
        model_report = json.loads(MODEL_REPORT_PATH.read_text(encoding="utf-8"))
        model_path = model_report.get("snapshot_path", "BAAI/bge-m3")
    else:
        model_path = "BAAI/bge-m3"

    model = BGEM3FlagModel(model_path, use_fp16=False, device="cpu")

    print("[INFO] Generating live dense & sparse query vectors...")
    encoded = model.encode(
        [query.query_text],
        batch_size=1,
        max_length=256,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False,
    )
    dense_vector = encoded["dense_vecs"][0].tolist()
    lex_weights = encoded["lexical_weights"][0]

    # Ταξινόμηση των sparse indices και values σε αύξουσα σειρά (απαίτηση του Qdrant)
    sorted_items = sorted((int(k), float(v)) for k, v in lex_weights.items())
    sorted_indices = [item[0] for item in sorted_items]
    sorted_values = [item[1] for item in sorted_items]

    sparse_vector = SparseVector(
        indices=sorted_indices,
        values=sorted_values,
    )

    # 3. Docker Qdrant Connection & Validation
    print(f"[INFO] Connecting to Docker Qdrant server at {args.qdrant_url}...")
    client = QdrantClient(url=args.qdrant_url, timeout=120)

    if SERVER_REPORT_PATH.exists():
        server_manifest = read_json_model(SERVER_REPORT_PATH, QdrantCollectionManifest)
        collection_name = server_manifest.collection_name
    else:
        collection_name = "rag_clerk_d984acfa6aae"

    if not client.collection_exists(collection_name):
        raise RuntimeError("Docker Qdrant collection does not exist.")

    point_count = client.count(collection_name=collection_name, exact=True).count
    if point_count != 5317:
        raise RuntimeError("Docker Qdrant collection is incomplete.")

    print(f"[INFO] Target Collection: {collection_name} (Points verified: {point_count})")

    # 4. Live Hybrid RRF Retrieval
    print("[INFO] Executing live Hybrid RRF Retrieval against Docker Qdrant...")
    retrieved_chunks = retrieve_hybrid_rrf(
        client=client,
        collection_name=collection_name,
        dense_vector=dense_vector,
        sparse_indices=sparse_vector.indices,
        sparse_values=sparse_vector.values,
        candidate_limit=args.candidate_limit,
        result_limit=args.candidate_limit,
    )

    ranked_doc_ids = rank_unique_documents(retrieved_chunks, limit=args.top_k)

    # Compute representation chunk per document
    representatives = {}
    for chunk in retrieved_chunks:
        doc_id = getattr(chunk, "doc_id", None)
        if doc_id and doc_id not in representatives:
            representatives[doc_id] = chunk

    # 5. Presentation of Top-K Results with Lineage
    print("-" * 60)
    print(f"LIVE RETRIEVAL RESULTS - TOP {args.top_k} (Split: {args.split}):")
    print("-" * 60)

    gold_doc_ids = {
        judgment.item_id
        for judgment in dataset.qrels.get(query.query_id, [])
        if getattr(judgment, "relevance", 0) > 0
    }

    for rank, doc_id in enumerate(ranked_doc_ids, start=1):
        chunk = representatives.get(doc_id)
        parent_id = getattr(chunk, "parent_id", "N/A") if chunk else "N/A"
        chunk_id = getattr(chunk, "chunk_id", "N/A") if chunk else "N/A"
        score = getattr(chunk, "score", 0.0) if chunk else 0.0
        text_snippet = getattr(chunk, "text", "")[:90].replace("\n", " ") if chunk else ""

        is_golden = doc_id in gold_doc_ids
        marker = "✅ [GOLDEN HIT]" if is_golden else "❌ [NO MATCH]"

        print(f"  {rank}. {marker}")
        print(f"     Doc ID      : {doc_id}")
        print(f"     Parent ID   : {parent_id}")
        print(f"     Chunk ID    : {chunk_id}")
        print(f"     Score       : {score:.4f}")
        if args.show_text and text_snippet:
            print(f"     Snippet     : {text_snippet}...")
        print()

    # 6. Live Ground-Truth Metrics Calculation
    retrieved_top_k = tuple(ranked_doc_ids)
    retrieved_gold_ids = gold_doc_ids.intersection(retrieved_top_k)
    hit_at_k = int(bool(retrieved_gold_ids))
    recall_at_k = len(retrieved_gold_ids) / len(gold_doc_ids) if gold_doc_ids else 0.0

    first_relevant_rank = next(
        (r for r, d in enumerate(retrieved_top_k, start=1) if d in gold_doc_ids),
        None,
    )
    reciprocal_rank = (
        1.0 / first_relevant_rank if first_relevant_rank is not None else 0.0
    )
    rel_rank_str = str(first_relevant_rank) if first_relevant_rank is not None else "None"

    print("-" * 60)
    print("LIVE GROUND-TRUTH CHECK")
    print("-" * 60)
    print(f"Golden document count        : {len(gold_doc_ids)}")
    print(f"Retrieved golden IDs found   : {len(retrieved_gold_ids)}")
    print(f"Hit@{args.top_k}                     : {hit_at_k}")
    print(f"Recall@{args.top_k}                  : {recall_at_k:.4f}")
    print(f"First relevant rank          : {rel_rank_str}")
    print(f"Reciprocal rank              : {reciprocal_rank:.4f}")
    print("==========================================================")

    client.close()


if __name__ == "__main__":
    main()
