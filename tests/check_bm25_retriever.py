from ingestion.document_loader import load_documents
from ingestion.text_chunker import chunk_documents

from retrieval.bm25_retriever import (
    build_bm25_index,
    retrieve
)

documents = load_documents()
chunks = chunk_documents(documents)

build_bm25_index(chunks)

results = retrieve(
    "What is LoRA?",
    top_k=5
)

for i, result in enumerate(results, start=1):
    print("=" * 80)
    print(f"Rank     : {i}")
    print(f"Score    : {result['score']:.4f}")
    print(f"Source   : {result['metadata']['source']}")
    print(f"Page     : {result['metadata']['page']}")
    print()
    print(result["text"][:300])
    print()