"""
Combines BM25 keyword retrieval and Chroma vector retrieval
using Reciprocal Rank Fusion (RRF).
"""


from retrieval.bm25_retriever import retrieve as bm25_retrieve
from retrieval.bm25_retriever import load_bm25_index
from retrieval.chroma_retriever import retrieve as chroma_retrieve

load_bm25_index()

def bm25_results(query,top_k=5):
    """
    Format BM25 search results.

    Args:
        query (str): User query.
        top_k (int): Number of chunks to retrieve.

    Returns:
        list: Formatted BM25 results.
    """
    results=bm25_retrieve(query,top_k)


    for rank,result in enumerate(results,start=1):
        result["id"]=result["metadata"]["chunk_id"]
        result["rank"]=rank
    return results


def chroma_results(query,top_k=5):
    """
    Format Chroma retrieval results.

    Args:
        query (str): User query.
        top_k (int): Number of chunks.

    Returns:
        list: Formatted Chroma results.
    """
    results=chroma_retrieve(query,top_k=5)

    formatted=[]

    docs=results["documents"][0]
    metas=results["metadatas"][0]

    for rank,(doc,meta) in enumerate(zip(docs,metas),start=1):

        formatted.append({
            "id": meta["chunk_id"],
            "text":doc,
            "metadata": meta,
            "rank": rank
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

        score=1/(k+result["rank"])
        fused_scores[result["id"]]=fused_scores.get(result["id"],0) + score

    for result in chroma_results:

        score=1/(k+result["rank"])
        fused_scores[result["id"]]=fused_scores.get(result["id"],0) + score


    return sorted(
        fused_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

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
   
    bm25_dict = {x["id"]: x for x in bm25}
    chroma_dict = {x["id"]: x for x in chroma}

    final_results = []



    for chunk_id, score in fused[:top_k]:

        chunk = chroma_dict.get(chunk_id)
        if chunk is None:
            chunk = bm25_dict.get(chunk_id)

        if chunk is None:
            continue

        chunk=chunk.copy()
        chunk["rrf_score"]= score

        final_results.append(chunk)

    return final_results



if __name__=="__main__":

    query="What is LoRA?"
    results=hybrid_search(query)
    for i,result in enumerate(results,start=1):
        print("="*80)
        print(f"Rank:",i)
        print("RRF Score:", result["rrf_score"])
        print(result["metadata"])
        print(result["text"][:500])

            
        