from retrieval.hybrid_retriever import hybrid_search

if __name__=="__main__":

    query="What is LoRA?"
    results=hybrid_search(query)
    for i,result in enumerate(results,start=1):
        print("="*80)
        print(f"Rank:",i)
        print("RRF Score:", result["rrf_score"])
        print(result["metadata"])
        print(result["text"][:500])