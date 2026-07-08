"""
Builds a BM25 index for keyword-based retrieval and
returns the most relevant document chunks for a query.
"""

from rank_bm25 import BM25Okapi
from ingestion.document_loader import load_documents
from ingestion.text_chunker import chunk_documents

# Global BM25 index
bm25= None

# Store document chunks used to build the index
chunks=None

def build_bm25_index(all_chunks):
    """
    Build a BM25 index from document chunks.
    Args:
        all_chunks(list):
            List of chunk dictionaries.
    """

    global bm25, chunks

    chunks=all_chunks
    
    # Tokenize chunk text
    tokenized_chunks=[
        chunk["text"].lower().split()
        for chunk in all_chunks
    ]
    
    # Build BM25 index
    bm25=BM25Okapi(tokenized_chunks)

    print(f"BM25 Index Built Successfully!")
    print(f"Indexed {len(all_chunks)} chunks")


def bm25_search(query,top_k=5):
    """
    Retrieve the most relevant chunks using BM25.
    Args:
        query(str):
            User query.

        top_k(int):
            Number of chunks to retrieve.
    Returns:
        list:
            Ranked list of (score,chunk)tuples.
    """
    tokenized_query=query.lower().split()

    scores=bm25.get_scores(tokenized_query)

    ranked_results=sorted(
        zip(scores,chunks),
        key=lambda x: x[0],
        reverse=True
    )
    
    return ranked_results[:top_k]
if __name__=="__main__":
    documents=load_documents()
    chunks=chunk_documents(documents)
    build_bm25_index(chunks)
    query="What is LoRA?"
    results=bm25_search(query)

    for rank,(score,chunk) in enumerate(results,start=1):
        print("="*80)
        print(f"Rank: {rank}")
        print(f"Score: {score:.4f}")
        print(chunk["metadata"])
        print(chunk["text"][:500])


