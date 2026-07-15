from vectorstore.chroma_store import (add_chunks_to_chroma,get_collection)
collection=get_collection()
if __name__=="__main__":

    if collection.count() >0:
        print(f"Collection already contains {collection.count()} vectors.")
        print("skipping ingestion.")
    else:
        documents=load_documents()
        chunks=chunk_documents(documents)
        embeddings=generate_embeddings(chunks)
        total_vectors=add_chunks_to_chroma(chunks,embeddings)
        print("Client Created Successfully")
        print(f"Collection Name: {collection.name}")
        print(f"Total Vectors : {total_vectors}")