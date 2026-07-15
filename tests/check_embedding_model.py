from ingestion.embedding_model import generate_embeddings

if __name__=="__main__":
    sample_chunks=[
        {
            "text":"Transformers use self attention."
        }
    ]
    
    embeddings=generate_embeddings(sample_chunks)
    print(type(embeddings))
    print()
    print("Embedding Shape:",embeddings.shape)
    print()
    print(embeddings[0][:10])