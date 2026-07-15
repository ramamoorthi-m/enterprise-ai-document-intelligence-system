"""
Retrieves the most relevant document chunks from ChromaDB
using semantic similarity search
"""


import chromadb
from vectorstore.chroma_store import get_collection
from ingestion.embedding_model import model



def retrieve(query,top_k=5):

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
    

  

    collection=get_collection()

    # Convert query into embedding vector 
    query_embedding=model.encode(query,normalize_embeddings=True).tolist()
  
    # Perform similarity search
    retrieval_results=collection.query(
     query_embeddings=[query_embedding],
     n_results=top_k,
     include=[
         "documents",
         "metadatas",
         "distances"
      ]
    )
    return retrieval_results

