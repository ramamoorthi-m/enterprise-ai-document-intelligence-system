from ingestion.document_loader import load_documents
from ingestion.text_chunker import chunk_documents

from retrieval.bm25_retriever import (
    build_bm25_index,
    save_bm25_index,
    load_bm25_index,
    retrieve
)


def main():

    print("=" * 70)
    print("BM25 Persistence Test")
    print("=" * 70)

    print("\nLoading documents...")
    documents = load_documents()

    print("Chunking documents...")
    chunks = chunk_documents(documents)

    print("Building BM25 index...")
    build_bm25_index(chunks)

    print("Saving BM25 index...")
    save_bm25_index()

    print("Loading BM25 index...")
    load_bm25_index()

    print("\nSearching...\n")

    results = retrieve(
        "Difference between GPT-3 and LLaMA",
        top_k=3
    )

    for i, result in enumerate(results, start=1):

        print("=" * 70)
        print(f"Rank   : {i}")
        print(f"Score  : {result['score']:.4f}")
        print(f"Source : {result['metadata']['source']}")
        print(f"Page   : {result['metadata']['page']}")
        print()
        print(result["text"][:300])
        print()


if __name__ == "__main__":
    main()