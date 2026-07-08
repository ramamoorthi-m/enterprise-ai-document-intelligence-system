"""
Combines BM25 keyword retrieval and Chroma vector retrieval
using Reciprocal Rank Fusion (RRF).
"""


from retrieval.bm25_retriever import bm25_search, build_bm25_index
from retrieval.chroma_retriever import retrieve
from ingestion.document_loader import load_documents
from ingestion.text_chunker import chunk_documents

def bm25_results(query,top_k=5):
    """
    Format BM25 search results.

    Args:
        query (str): User query.
        top_k (int): Number of chunks to retrieve.

    Returns:
        list: Formatted BM25 results.
    """
    results=bm25_search(query,top_k=5)

    formatted=[]

    for rank,(score,chunk) in enumerate(results,start=1):
        formatted.append({
            "id": chunk["metadata"]["chunk_id"],
            "text":chunk["text"],
            "metadata": chunk["metadata"],
            "rank": rank
        })
    return formatted


def chroma_results(query,top_k=5):
    """
    Format Chroma retrieval results.

    Args:
        query (str): User query.
        top_k (int): Number of chunks.

    Returns:
        list: Formatted Chroma results.
    """
    results=retrieve(query,top_k=5)

    formatted=[]

    docs=results["documents"][0]
    metas=results["metadatas"][0]

    for rank in range(len(docs)):
        formatted.append({
            "id": metas[rank]["chunk_id"],
            "text":docs[rank],
            "metadata": metas[rank],
            "rank": rank+1
        })
    return formatted

def reciprocal_rank_fusion(bm25_results,chroma_results,k=60):
    """
    Fuse BM25 and Chroma rankings using Reciprocal Rank Fusion.

    Args:
        bm25_results (list): BM25 ranked results.
        chroma_results (list): Chroma ranked results.
        k (int): RRF constant.

    Returns:
        list: Sorted (chunk_id, score) pairs.
    """

    fused_scores={}
    for result in bm25_results:

        chunk_id=result["id"]
        rank=result["rank"]

        score=1/(k+rank)

        fused_scores[chunk_id]=score

    for result in chroma_results:

        chunk_id=result["id"]
        rank=result["rank"]

        score=1/(k+rank)

        if chunk_id in fused_scores:
            fused_scores[chunk_id]+=score
        else:
            fused_scores[chunk_id]=score
    sorted_results=sorted(
        fused_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )
    return sorted_results

def hybrid_search(query, top_k=5):
    """
    Perform hybrid retrieval using BM25 + Chroma + RRF.

    Args:
        query (str): User query.
        top_k (int): Number of final chunks.

    Returns:
        list: Final fused retrieval results.
    """


    bm25 = bm25_results(query, top_k)
    chroma = chroma_results(query, top_k)

    fused = reciprocal_rank_fusion(bm25, chroma)
    final_results = []


    bm25_dict = {chunk["id"]: chunk for chunk in bm25}
    chroma_dict = {chunk["id"]: chunk for chunk in chroma}

    for chunk_id, score in fused[:top_k]:

        if chunk_id in chroma_dict:
            chunk = chroma_dict[chunk_id]
        else:
            chunk = bm25_dict[chunk_id]

        chunk["rrf_score"]= score
        final_results.append(chunk)

    return final_results



if __name__=="__main__":
    docs=load_documents()
    chunks=chunk_documents(docs)
    build_bm25_index(chunks)

    query="What is LoRA?"
    results=hybrid_search(query)
    for i,chunk in enumerate(results,start=1):
        print("="*80)
        print(f"Rank:",i)
        print("RRF Score:", chunk["rrf_score"])
        print(result["metadata"])
        print(result["text"][:500])

            
        