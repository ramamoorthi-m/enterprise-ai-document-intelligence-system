"""
Extracts unique document citations from the retrieved chunks.
"""


def extract_sources(retrieved_chunks):
    """
    Extract unique document sources.

    Args:
        retrieved_chunks (list):
            Retrieved chunks containing metadata.

    Returns:
        List[str]:
            List of unique source citations.
    """


    seen=set()
    sources=[]

    for chunk in retrieved_chunks:

        metadata=chunk["metadata"]

        source=metadata.get("source","Unknown")
        page=metadata.get("page","Unknown")

        key=(source,page)

        if key not in seen:
            seen.add(key)
            sources.append(f"{source} - Page{page}")

    return sources