"""
Loads the sentence-transformer embedding model and converts
text chunks into dense vector embeddings for semantic search.
"""

from sentence_transformers import SentenceTransformer

model=SentenceTransformer("BAAI/bge-base-en-v1.5")

def generate_embeddings(chunks):
    """
    Generate vector embedddings for document chunks.
    Args:
        chunks(list):
            list of chunk dictionaries.
            Each chunk must contain a "text" field:
    Returns:
        numpy.ndarray:
            Array containing one embedding vector per chunk.
    """

    texts=[chunk["text"] for chunk in chunks]
    if not chunks:
        raise ValueError("No chunks provided for  embedding generation.")
    embeddings = model.encode(texts,convert_to_numpy=True,show_progress_bar=False)
    return embeddings

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