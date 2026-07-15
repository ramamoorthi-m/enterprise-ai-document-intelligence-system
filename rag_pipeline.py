from retrieval.bm25_retriever import load_bm25_index
from retrieval.chroma_retriever import get_collection

from retrieval.hybrid_retriever import hybrid_search
from retrieval.reranker import rerank

from generation.query_rewriter import rewrite_query
from generation.llm_generator import generate_answer
from generation.prompt_builder import build_prompt
from generation.source_citation import extract_sources

from memory.chat_memory import ChatMemory


def should_rewrite_query(query: str, history):
    """
    Rewrite only follow-up questions.
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

    return (
        len(words & pronouns) > 0 or
        len(words & comparison_words) > 0 or
        len(words & followup_words) > 0
    )

# Global conversation memory
memory=ChatMemory()

def initialize():
    print("=" * 60)
    print("Initializing Retrieval System...")
    print("=" * 60)

    print("Loading ChromaDB...")
    get_collection()

    print("Loading BM25 index...")
    load_bm25_index()

    print("Initialization complete.")

def ask(query,return_contexts=False):

    history=memory.get_history()
    rewritten_query=query

    try:
        if should_rewrite_query(query,history):

            print("\nQuery Rewriter : ON")

            rewritten_query=rewrite_query(query,history)

            if rewritten_query.lower() in [
                "answer",
                "summary",
                "explanation"
                ]:
                  rewritten_query=query

        else:
             print("\nQuery Rewriter : OFF")

    except Exception as e:
        print(f"Query rewrite failed: {e}")
        rewritten_query

    print("=" * 80)
    print(f"Original Query : {query}")
    print(f"Final Query : {rewritten_query}")
    print("=" * 80)


    # Retrieve candidate chunks
    retrieved_chunks=hybrid_search(rewritten_query,top_k=20)

    # Rerank retrieved chunks
    retrieved_chunks=rerank(rewritten_query,retrieved_chunks,top_k=5)

    # Build LLM prompt
    prompt=build_prompt(rewritten_query,retrieved_chunks,history)

    # Generate Answer
    try:
        answer=generate_answer(prompt)
    except Exception as e:
        print("LLM Error:",e)
        answer="LLM is currently unavailable"
    # Save conversation
    memory.add(query,answer)

    # Extract document citations
    contexts=[chunk["text"] for chunk in retrieved_chunks]
    sources=extract_sources(retrieved_chunks)

    if return_contexts:
        return answer,contexts,sources


    return answer,contexts,sources

if __name__=="__main__":
    initialize()
    while True:
       query=input("\nAsk Question (type 'exit to quit):")
       if query.lower()=="exit":
        break
       answer,contexts,sources=ask(query)
       print("\n" + "=" * 80)
       print("\nAnswer:\n")
       print(answer)

       print("\nSources:")
       for source in sources:
        print(f"- {source}")
       print("=" * 80)


              