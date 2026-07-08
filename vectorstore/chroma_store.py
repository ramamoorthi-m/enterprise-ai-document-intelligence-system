"""
Creates and manages the ChromaDB vector store for the RAG pipeline.
Stores document chunks,metadata,and embeddings for semantic retrieval.
"""


import chromadb
from ingestion.document_loader import load_documents
from ingestion.text_chunker import chunk_documents
from ingestion.embedding_model import generate_embeddings


# Initialize ChromaDB client
client=chromadb.PersistentClient(
    path="chroma_db"
)

#Create or load the collection
collection=client.get_or_create_collection(
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

if __name__=="__main__":
    if collection.count() >0:
        print(f"Collection already contains {collection.count()} vectors.")
        print("skipping ingestion.")
    else:
        documents=load_documents()
        chunks=chunk_documents(documents)
        embeddings=generate_embeddings(chunks)
        add_chunks_to_chroma(chunks,embeddings)
        print("Client Created Successfully")
        print("Collection Name:", collection.name)

        

  

