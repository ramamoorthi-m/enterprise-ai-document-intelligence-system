from generation.query_rewriter import rewrite_query

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
