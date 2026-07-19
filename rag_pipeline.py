import time

from retrieval.bm25_retriever import load_bm25_index
from retrieval.chroma_retriever import get_collection
from retrieval.hybrid_retriever import hybrid_search
from retrieval.reranker import rerank

from generation.query_rewriter import rewrite_query
from generation.prompt_builder import build_prompt
from generation.llm_generator import generate_answer
from generation.source_citation import extract_sources

from memory.chat_memory import ChatMemory


###############################################################################
# Configuration
###############################################################################

INITIAL_RETRIEVAL_K = 20
FINAL_RERANK_K = 5

memory = ChatMemory()


###############################################################################
# Query Rewrite Decision
###############################################################################

def should_rewrite_query(query: str, history: str) -> bool:
    """
    Decide whether the query should be rewritten.
    Only rewrite follow-up questions.
    """

    if not history.strip():
        return False

    query = query.lower()

    pronouns = {
        "it", "its",
        "they", "them", "their",
        "this", "that",
        "these", "those",
        "he", "she", "him", "her"
    }

    comparison_words = {
        "compare",
        "difference",
        "same",
        "previous",
        "above",
        "former",
        "latter"
    }

    followup_words = {
        "why",
        "how",
        "when",
        "where",
        "more",
        "explain",
        "advantages",
        "disadvantages",
        "benefits",
        "drawbacks",
        "pros",
        "cons"
    }

    words = set(query.split())

    return bool(
        words & pronouns or
        words & comparison_words or
        words & followup_words
    )


###############################################################################
# Initialization
###############################################################################

def initialize():

    print("=" * 70)
    print("Initializing Enterprise Retrieval System...")
    print("=" * 70)

    get_collection()
    print(" ChromaDB Loaded")

    load_bm25_index()
    print(" BM25 Loaded")

    print(" System Ready")
    print("=" * 70)


###############################################################################
# Main Pipeline
###############################################################################

def ask(query: str, return_contexts=False):

    start = time.time()

    history = memory.get_history()

    rewritten_query = query

    ##############################################################
    # Query Rewrite
    ##############################################################

    try:

        if should_rewrite_query(query, history):

            print("\nQuery Rewriter : ON")

            rewritten_query = rewrite_query(query, history)

            if rewritten_query.lower() in {
                "answer",
                "summary",
                "explanation",
                ""
            }:
                rewritten_query = query

        else:
            print("\nQuery Rewriter : OFF")

    except Exception as e:

        print("Query Rewrite Error:", e)
        rewritten_query = query

    print("=" * 80)
    print("Original Query :", query)
    print("Final Query    :", rewritten_query)
    print("=" * 80)

    ##############################################################
    # Hybrid Retrieval
    ##############################################################

    retrieved_chunks = hybrid_search(
        rewritten_query,
        top_k=INITIAL_RETRIEVAL_K
    )

    if not retrieved_chunks:

        return (
            "No relevant documents found.",
            [],
            []
        )

    ##############################################################
    # Reranking
    ##############################################################

    retrieved_chunks = rerank(
        rewritten_query,
        retrieved_chunks,
        top_k=FINAL_RERANK_K
    )

    ##############################################################
    # Prompt Building
    ##############################################################

    prompt = build_prompt(
        rewritten_query,
        retrieved_chunks,
        history
    )

    ##############################################################
    # LLM Generation
    ##############################################################

    try:

        answer = generate_answer(prompt)

    except Exception as e:
        print("=" * 80)
        print("LLM ERROR")
        print(type(e))
        print(str(e))
        print("="* 80)

        return f"LLM Error: {str(e)}", [], []

    ##############################################################
    # Save Conversation
    ##############################################################

    memory.add(query, answer)

    ##############################################################
    # Sources
    ##############################################################

    contexts = [
        chunk["text"]
        for chunk in retrieved_chunks
    ]

    sources = extract_sources(retrieved_chunks)

    ##############################################################
    # Metrics
    ##############################################################

    elapsed = round(time.time() - start, 2)

    print(f"\nPipeline Time : {elapsed} sec")
    print("=" * 80)

    if return_contexts:
        return answer, contexts, sources

    return answer, contexts, sources
