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

    # Combine retrieved chunks into a single context
    for chunk in retrieved_chunks:
        context +=chunk["text"]+ "\n\n"

    prompt=f""" 
You are an AI assistant for enterprise document intelligence that answers questions
using only the provided context.


If the answer cannot be found in the context,
reply:

"I could not find the answer in the provided documents."

Do not make up information.
Do not use outside knowledge

Previous Conversation:
{history}

Context:
{context}

Question:
{query}

Answer:
"""
    return prompt
    

    



    



if __name__=="__main__":
    dummy_results={
        "documents":[[
            "Self attention allows a model to focus on relevant words.",
            "Transformer uses multi-head attention."
        ]],
        "metadatas":[[
            {
                "source":
                "attention_is_all_you_need.pdf",
                "page": 4
            },
            {
                "source":
                "attention_is_all_you_need.pdf",
                "page": 5
            }


        ]]
        
    }
    prompt=build_prompt(
        "What is self attention?",
        dummy_results
    )
    print(prompt)