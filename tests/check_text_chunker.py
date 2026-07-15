from ingestion.text_chunker import chunk_documents

if __name__=="__main__":
    sample_documents=[
        {
            "text":(
                "Artificial Intelligence is transforming industries."
                "Machine Learning allows computers to learn from data."
                "Deep learning is a subset of machine learning."
                "Large Language Models are changing how humans interact with AI."
                ),
                "source":"sample.pdf",
                "page":1
                
        }

    ]
    chunks=chunk_documents(sample_documents)
    print(f"\nTotal Chunks: {len(chunks)}\n")
    for chunk in chunks:
        print("="*80)
        print(chunk["metadata"])
        print(chunk["text"])
