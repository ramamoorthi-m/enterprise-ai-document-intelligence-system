"""
Rewrites follow-up user questions into standalone search queries
to improve retrieval quality in the RAG pipeline.
"""
from groq import Groq 
from dotenv import load_dotenv
import os

load_dotenv()

client=Groq(api_key=os.getenv("GROQ_API_KEY"))

def rewrite_query(query,history):
    prompt=f"""
Conversation History:
{history}

Current User Question:
{query}

Rules:

1.If the  current question depends on previous conversation,
rewrite into a  complete standalone question.

2.If the current question is already standalone, RETURN IT EXACTLY AS IT IS.

3.Do NOT improve the wording.

4.Do NOT add new information.

5.Do NOT explain.

6.Do NOT answer.

7.Preserve the user's original intent exactly.

Return ONLY the final rewritten question.
"""
    
    response=client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role":"user",
                "content":prompt
            }
        ],
        temperature=0
    )
    return response.choices[0].message.content.strip()

if __name__=="__main__":
    history= """
    User: What is LoRA?
Assistant: LoRA stands for Low-Rank Adaptation.
It is a parameter-efficient fine-tuning technique.
"""


    query="What is LoRA?"
    rewritten=rewrite_query(query,history)

    print("=" * 60)
    print("Original Query:")
    print(query)
    print("=" * 60)
    print("Rewritten Query:")
    print(rewritten)
    print("=" * 60)
