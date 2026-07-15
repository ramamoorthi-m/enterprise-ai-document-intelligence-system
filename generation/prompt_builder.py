"""
Builds the final prompt that is sent to the LLM.
The prompt contains conversation history, retrieved
context, and the user's query.
"""


def build_prompt(query, retrieved_chunks,history=""):
    """
    Build the prompt for the language model.

    Args:
        query (str):
            User's question.

        retrieved_chunks (list):
            Retrieved document chunks.

        history (str):
            Previous conversation history.

    Returns:
        str:
            Final prompt for the LLM.
    """

    context= ""

    for i,chunk in enumerate(retrieved_chunks,start=1):
        meta=chunk.get("metadata",{})

        context +=(
            f"[Document {i}]\n"
            f"Source : {meta.get('source','Unknown')}\n"
            f"Page  : {meta.get('page','-')}\n\n"
            f"{chunk['text']}\n\n"
        )

    prompt=f""" 
You are an Enterprise AI assistant.

Instructions:
-Answer ONLY from the provided context.
-If the answer is not present,say:
 "I could not find the answer in the provided documents."
-Do not use outside knowledge.
-Do not hallucinate.
-Always cite the source document and page number whenever possible.
-keep the answer concise and accurate. 

Previous Conversation:
{history}

Context:
{context}

Question:
{query}

Answer:
"""
    return prompt
    

    



    



