from retrieval.chroma_retriever import retrieve

if __name__=="__main__":
    query="Difference between GPT-3 and LLaMA?"
    results=retrieve(query)

    documents=results["documents"][0]
    metadatas=results["metadatas"][0]
    distances=results["distances"][0]

    for i in range(len(documents)):
        print("="*80)
        print("Rank:",i+1)
        print("Distance:", distances[i])
        print("Source:", metadatas[i]["source"])
        print("Page:", metadatas[i]["page"])
        print()
        print(documents[i][:300] + "...")
        
    