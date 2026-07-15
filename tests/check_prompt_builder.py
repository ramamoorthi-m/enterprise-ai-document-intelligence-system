from generation.prompt_builder import build_prompt

if __name__ == "__main__":

    dummy_chunks = [
        {
            "text": "Self attention allows a model to focus on relevant words.",
            "metadata": {
                "source": "attention_is_all_you_need.pdf",
                "page": 4
            }
        },
        {
            "text": "Transformer uses multi-head attention.",
            "metadata": {
                "source": "attention_is_all_you_need.pdf",
                "page": 5
            }
        }
    ]

    prompt = build_prompt(
        "What is self attention?",
        dummy_chunks
    )

    print(prompt)