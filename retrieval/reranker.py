"""
Re-ranks retrieved chunks using a CrossEncoder model
to improve retrieval accuracy before LLM generation.
"""

from sentence_transformers import CrossEncoder

# CrossEncoder model for semantic reranking
reranker=CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank(query,retrieved_chunks,top_k=5):
    """
    Re-rank retrieved chunks based on semantic relevance.

    Args:
        query (str):
            User query.

        retrieved_chunks (list):
            Retrieved chunks from Hybrid Retriever.

        top_k (int):
            Number of final chunks to return.

    Returns:
        list:
            Re-ranked chunks sorted by semantic relevance.
    """
    if not retrieved_chunks:
        return []

    # Create(query,chunk) pairs
    pairs=[
        (query,chunk["text"])
        for chunk in retrieved_chunks

    ]

    # Predict relevance scores
    scores=reranker.predict(pairs, batch_size=16, show_progress_bar=False)

    # Attach rerank score
    reranked=[]
    for chunk,score in zip(retrieved_chunks,scores):
        item=chunk.copy()
        item["rerank_score"]=float(score)
        reranked.append(item)

    # Sort by reranker score
    reranked=sorted(
        reranked,
        key=lambda x: x["rerank_score"],
        reverse=True
    )

    return reranked[:top_k]