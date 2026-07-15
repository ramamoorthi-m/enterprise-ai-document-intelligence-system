"""
BM25 Retriever

Provides keyword-based retrieval using the BM25 ranking algorithm.
The BM25 index is built during offline ingestion and initialized once
when the backend starts.
"""

from rank_bm25 import BM25Okapi
import numpy as np
import pickle
from pathlib import Path

# Global objects
_bm25 = None
_chunks = None

BM25_INDEX_PATH = Path("retrieval")/"bm25_index.pkl"


def build_bm25_index(all_chunks):
    """
    Build the BM25 index from document chunks.

    Args:
        all_chunks (list): List of chunk dictionaries.

    Returns:
        BM25Okapi
    """

    global _bm25, _chunks

    if not all_chunks:
        raise ValueError("No chunks provided to build BM25 index.")

    _chunks = all_chunks

    tokenized_chunks = [
        chunk["text"].lower().split()
        for chunk in all_chunks
        if chunk.get("text", "").strip()
    ]

    if len(tokenized_chunks) != len(all_chunks):
        raise ValueError("Some chunks contain empty text.")

    _bm25 = BM25Okapi(tokenized_chunks)

    print(f"BM25 index built successfully.")
    print(f"Indexed {len(_chunks)} chunks.")

    return _bm25

def save_bm25_index():
    """
    Save the BM25 index and chunks to disk.
    """

    if _bm25 is None or _chunks is None:
        raise RuntimeError(
            "BM25 index has not been built."
        )

    data = {
        "bm25": _bm25,
        "chunks": _chunks
    }

    with open(BM25_INDEX_PATH, "wb") as file:
        pickle.dump(data, file)

    print(f"BM25 index saved successfully.")
    print(f"Location : {BM25_INDEX_PATH}")


def load_bm25_index():
    """
    Load the BM25 index and chunks from disk.
    """

    global _bm25, _chunks

    if not BM25_INDEX_PATH.exists():
        raise FileNotFoundError(
            f"{BM25_INDEX_PATH} does not exist."
        )

    with open(BM25_INDEX_PATH, "rb") as file:
        data = pickle.load(file)

    _bm25 = data["bm25"]
    _chunks = data["chunks"]

    print("BM25 index loaded successfully.")

    return _bm25


def retrieve(query, top_k=5):
    """
    Retrieve the most relevant chunks using BM25.

    Args:
        query (str): User query.
        top_k (int): Number of chunks to retrieve.

    Returns:
        list
    """

    if _bm25 is None:
        raise RuntimeError(
            "BM25 index has not been initialized."
        )

    tokenized_query = query.lower().split()

    scores = _bm25.get_scores(tokenized_query)

    top_indices = np.argsort(scores)[::-1][:top_k]

    retrieval_results = []

    for idx in top_indices:
        retrieval_results.append({
            "score": float(scores[idx]),
            "text": _chunks[idx]["text"],
            "metadata": _chunks[idx]["metadata"]
        })

    return retrieval_results


def is_initialized():
    """
    Check whether the BM25 index has been built.
    """

    return _bm25 is not None

