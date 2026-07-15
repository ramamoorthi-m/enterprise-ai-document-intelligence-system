from ingestion.document_loader import load_documents
from ingestion.text_chunker import chunk_documents
from ingestion.embedding_model import generate_embeddings
from vectorstore.chroma_store import add_chunks_to_chroma
from retrieval.bm25_retriever import (build_bm25_index,save_bm25_index)

def build_knowledge_base():
    print("\nLoading documents..")
    documents=load_documents()
    print(f"Loaded {len(documents)} pages")

    print("\nCreating chunks...")
    chunks=chunk_documents(documents)
    print(f"Created {len(chunks)} chunks")

    print("\nGenerating Embeddings...")
    embeddings=generate_embeddings(chunks)
    print(f"Generated {len(embeddings)} embeddings")

    print("\nBuilding  chroma knowledge base...")
    total_vectors=add_chunks_to_chroma(chunks,embeddings)
    print(f"Stored {total_vectors} vectors in ChromaDB")

    print("\nBuilding BM25 index...")
    build_bm25_index(chunks)

    print("Saving BM25 index...")
    save_bm25_index()
    
    print("\nKnowledge base built successfully.")