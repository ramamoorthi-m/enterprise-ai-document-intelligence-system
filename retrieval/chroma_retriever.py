"""
Retrieves the most relevant document chunks from ChromaDB
using semantic similarity search
"""


import chromadb
from ingestion.embedding_model import model

# Initialize ChromaDB client
client=chromadb.PersistentClient(
    path="chroma_db"
)

# Load existing collection
collection=client.get_or_create_collection(
    name="enterprise_ai"
)


def retrieve(
    query,
    top_k=5
):

  """
  Retrieve the most relevant document chunks.
  Args:
       query(str):
           User query,
        top_k(int):
        Number of similar chunks to retrieve.

    Returns:
        dict:
            ChromaDB query results.
    """

  # Convert query into embedding vector 
  query_embedding=model.encode(query).tolist()
  
  # Perform similarity search
  results=collection.query(
    query_embeddings=[query_embedding],
    n_results=top_k
  )
  return results

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
        print(documents[i])
        
    