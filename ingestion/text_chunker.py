"""
Text chunking utilities for document ingestion.
This module splits extracted documents into overlapping chunks
while preserving metadata required for retrieval.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
text_splitter=RecursiveCharacterTextSplitter(
    chunk_size=700,
    chunk_overlap=150,
    separators=[
        "\n\n",
        "\n",
        ".",
        " ",
        ""
    ]
)

def chunk_documents(documents):
    """
    Split documents into smaller overlapping chunks.
    Args:
         documents(list[dict]):
          List of extracted documents.
    Returns:
            list[dict]:
                  List of chunk dictionaries with metadata.
    """

    chunks=[]
    chunk_id=0

    for document in  documents:
        text=document["text"]
        split_texts=text_splitter.split_text(text)

        for chunk in split_texts:

            chunks.append(
                {
                    "text":chunk,
                    "metadata":{
                        "source":document["source"],
                        "page":document["page"],
                        "chunk_id":chunk_id
                    }
                }
            )
            chunk_id +=1
            
    return chunks

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
