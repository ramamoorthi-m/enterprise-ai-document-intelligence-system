"""
Creates and manages the ChromaDB vector store for the RAG pipeline.
Stores document chunks,metadata,and embeddings for semantic retrieval.
"""


import chromadb

# Initialize ChromaDB client
client=chromadb.PersistentClient(
    path="chroma_db"
)

#
def get_collection():
    """
    Create or load the collection
    """
    return client.get_or_create_collection(
    name="enterprise_ai"
)

def add_chunks_to_chroma(chunks,embeddings):
    """
    Store document chunks and their embeddings in ChromaDB.
    Args:
        chunks(list):
            List of chunk dictionaries.
        embeddings:
            NumPy array containing embedding vectors.
        """

    if len(chunks) !=len(embeddings):
        raise ValueError(
            "Number of chunks and embeddings must be equal"
        )

    collection=get_collection()

    # Create unique IDs for each chunk
    ids=[
        str(chunk["metadata"]["chunk_id"])
        for chunk in chunks
    ]

    # Extract chunk text
    documents=[
        chunk["text"]
        for chunk in chunks
    ]
    
    # Extract metadata
    metadatas=[
        chunk["metadata"]
        for chunk in chunks
    ]
    
    # Store vectors,documents and metadata
    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings.tolist(),
        metadatas=metadatas
        )

    print(f"\nStored {len(chunks)} chunks successfully.")

    return collection.count()



        

  

